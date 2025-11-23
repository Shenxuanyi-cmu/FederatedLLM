from peft import set_peft_model_state_dict
import torch
import os
from torch.nn.functional import normalize
from torch.nn import ZeroPad2d


def FedAvg(
    model,
    selected_clients_set,
    output_dir,
    local_dataset_len_dict,
    epoch,
    stacking,
    lora_r,
    heter,
    local_ranks,
    zero_padding,
    full,
    freezeA_phase=False,      # ⭐ Freeze-A toggle
):

    # ===========================================================
    # Compute client weights
    # ===========================================================
    weights_array = normalize(
        torch.tensor([local_dataset_len_dict[cid] for cid in selected_clients_set],
                     dtype=torch.float32),
        p=1, dim=0
    )
    print("Weights:", weights_array)

    # global weights for Freeze-A
    global_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # ===========================================================
    # Aggregate client models
    # ===========================================================
    for k, cid in enumerate(selected_clients_set):

        single_output_dir = os.path.join(
            output_dir, str(epoch), f"local_output_{cid}", "pytorch_model.bin"
        )
        single_weights = torch.load(single_output_dir, map_location="cpu")

        # =======================================================
        # ⭐ Freeze-A: keep A matrices unchanged
        # =======================================================
        if freezeA_phase:
            for name in list(single_weights.keys()):
                if "lora_A" in name:
                    single_weights[name] = global_state[name].clone()

        # =======================================================
        # full model aggregation
        # =======================================================
        if full:
            if k == 0:
                weighted_single_weights = {
                    key: single_weights[key] * weights_array[k]
                    for key in single_weights.keys()
                }
            else:
                for key in single_weights.keys():
                    weighted_single_weights[key] += single_weights[key] * weights_array[k]

        # =======================================================
        # LoRA modes
        # =======================================================
        else:
            # -----------------------------
            # Stacking variants
            # -----------------------------
            if stacking:

                # ----- stacking + zero padding -----
                if zero_padding:
                    max_lora = max(local_ranks)

                    if k == 0:
                        weighted_single_weights = {}
                        for key in single_weights.keys():
                            w = single_weights[key]

                            if w.dim() == 2:
                                if w.shape[0] == local_ranks[cid]:
                                    w = ZeroPad2d((0, 0, 0, max_lora - local_ranks[cid]))(w)
                                elif w.shape[1] == local_ranks[cid]:
                                    w = ZeroPad2d((0, max_lora - local_ranks[cid], 0, 0))(w)

                            weighted_single_weights[key] = w * weights_array[k]

                    else:
                        for key in single_weights.keys():
                            w = single_weights[key]

                            if w.dim() == 2:
                                if w.shape[0] == local_ranks[cid]:
                                    w = ZeroPad2d((0, 0, 0, max_lora - local_ranks[cid]))(w)
                                elif w.shape[1] == local_ranks[cid]:
                                    w = ZeroPad2d((0, max_lora - local_ranks[cid], 0, 0))(w)

                            weighted_single_weights[key] += w * weights_array[k]

                # ----- stacking without padding -----
                else:
                    if k == 0:
                        weighted_single_weights = {}

                        for key in single_weights.keys():
                            w = single_weights[key]

                            if heter:
                                if w.shape[0] == local_ranks[cid]:
                                    weighted_single_weights[key] = w * weights_array[k]
                                else:
                                    weighted_single_weights[key] = w
                            else:
                                if w.shape[0] == lora_r:
                                    weighted_single_weights[key] = w * weights_array[k]
                                else:
                                    weighted_single_weights[key] = w

                    else:
                        for key in single_weights.keys():
                            w = single_weights[key]

                            if heter:
                                if w.shape[0] == local_ranks[cid]:
                                    weighted_single_weights[key] = torch.cat(
                                        [weighted_single_weights[key], w * weights_array[k]], dim=0
                                    )
                                else:
                                    weighted_single_weights[key] = torch.cat(
                                        [weighted_single_weights[key], w], dim=1
                                    )
                            else:
                                if w.shape[0] == lora_r:
                                    weighted_single_weights[key] = torch.cat(
                                        [weighted_single_weights[key], w * weights_array[k]], dim=0
                                    )
                                else:
                                    weighted_single_weights[key] = torch.cat(
                                        [weighted_single_weights[key], w], dim=1
                                    )

            # =======================================================
            # Vanilla LoRA FedAvg (no stacking)
            # =======================================================
            else:
                if zero_padding:
                    max_lora = max(local_ranks)

                    if k == 0:
                        weighted_single_weights = {}

                        for key in single_weights.keys():
                            w = single_weights[key]
                            if w.dim() == 2:
                                if w.shape[0] == local_ranks[cid]:
                                    w = ZeroPad2d((0, 0, 0, max_lora - local_ranks[cid]))(w)
                                elif w.shape[1] == local_ranks[cid]:
                                    w = ZeroPad2d((0, max_lora - local_ranks[cid], 0, 0))(w)

                            weighted_single_weights[key] = w * weights_array[k]

                    else:
                        for key in single_weights.keys():
                            w = single_weights[key]
                            if w.dim() == 2:
                                if w.shape[0] == local_ranks[cid]:
                                    w = ZeroPad2d((0, 0, 0, max_lora - local_ranks[cid]))(w)
                                elif w.shape[1] == local_ranks[cid]:
                                    w = ZeroPad2d((0, max_lora - local_ranks[cid], 0, 0))(w)

                            weighted_single_weights[key] += w * weights_array[k]

                else:
                    if k == 0:
                        weighted_single_weights = {
                            key: single_weights[key] * weights_array[k]
                            for key in single_weights.keys()
                        }
                    else:
                        for key in single_weights.keys():
                            weighted_single_weights[key] += single_weights[key] * weights_array[k]

    # ===========================================================
    # Apply merged weights
    # ===========================================================
    if stacking:
        torch.save(weighted_single_weights, os.path.join(output_dir, str(epoch), "adapter_model.bin"))
        return model

    elif full:
        torch.save(weighted_single_weights, os.path.join(output_dir, str(epoch), "pytorch_model.bin"))
        model.load_state_dict(weighted_single_weights)
        return model

    else:
        set_peft_model_state_dict(model, weighted_single_weights, "default")
        return model
