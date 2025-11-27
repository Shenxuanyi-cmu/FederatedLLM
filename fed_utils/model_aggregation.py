from peft import set_peft_model_state_dict
import torch
import os
from torch.nn.functional import normalize
from torch.nn import ZeroPad2d


# utility for checking is LoRA weight
def is_lora_A(name):
    return ("lora_A" in name) and ("default" in name)


def is_lora_B(name):
    return ("lora_B" in name) and ("default" in name)


# ===========================================================
#  FedAvg / Stacking / Freeze-A
# ===========================================================
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
    freezeA_phase=False     # ⭐ FREEZE-A FLAG
):
    # -----------------------------------------------------------
    # Compute FedAvg weights ∝ client dataset length
    # -----------------------------------------------------------
    size_tensor = torch.tensor(
        [local_dataset_len_dict[cid] for cid in selected_clients_set],
        dtype=torch.float32
    )
    weights_array = normalize(size_tensor, p=1, dim=0)
    print("Weights:", weights_array)

    # -----------------------------------------------------------
    # Load global (original) LoRA A, used when Freezing-A
    # -----------------------------------------------------------
    global_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # -----------------------------------------------------------
    # Begin aggregation
    # -----------------------------------------------------------
    weighted_single_weights = None

    for idx, cid in enumerate(selected_clients_set):

        single_path = os.path.join(
            output_dir, str(epoch),
            f"local_output_{cid}", "pytorch_model.bin"
        )
        single_weights = torch.load(single_path, map_location="cpu")

        # ================================
        # FREEZE-A: replace A_i with global A
        # ================================
        if freezeA_phase:
            for name in single_weights.keys():
                if is_lora_A(name):
                    single_weights[name] = global_state[name].clone()

        # ================================
        # FULL MODEL FedAvg
        # ================================
        if full:
            if idx == 0:
                weighted_single_weights = {
                    k: v * weights_array[idx]
                    for k, v in single_weights.items()
                }
            else:
                for k in single_weights.keys():
                    weighted_single_weights[k] += single_weights[k] * weights_array[idx]
            continue

        # =======================================================
        # NOT full: LoRA-only aggregation
        # stacking vs FedAvg
        # =======================================================
        if stacking:
            # ----------------------------------------------------
            # stacking mode (concat B; A only if not freezeA)
            # ----------------------------------------------------
            if zero_padding:
                max_r = max(local_ranks)

                # ------------------------------------
                # first client
                # ------------------------------------
                if idx == 0:
                    weighted_single_weights = {}

                    for name, tensor in single_weights.items():

                        # LoRA A
                        if is_lora_A(name):
                            if freezeA_phase:
                                weighted_single_weights[name] = global_state[name]
                            else:
                                # pad A to max_r
                                w = tensor
                                if w.dim() == 2 and w.shape[0] < max_r:
                                    pad = ZeroPad2d((0, 0, 0, max_r - w.shape[0]))
                                    w = pad(w)
                                weighted_single_weights[name] = w
                            continue

                        # LoRA B
                        if is_lora_B(name):
                            w = tensor * weights_array[idx]
                            if w.dim() == 2 and w.shape[1] < max_r:
                                pad = ZeroPad2d((0, max_r - w.shape[1], 0, 0))
                                w = pad(w)
                            weighted_single_weights[name] = w
                            continue

                        weighted_single_weights[name] = tensor
                # ------------------------------------
                # next clients
                # ------------------------------------
                else:
                    for name, tensor in single_weights.items():

                        if is_lora_A(name):
                            continue     # A 不拼接，也不加权

                        if is_lora_B(name):
                            w = tensor * weights_array[idx]

                            if w.dim() == 2 and w.shape[1] < max_r:
                                pad = ZeroPad2d((0, max_r - w.shape[1], 0, 0))
                                w = pad(w)

                            weighted_single_weights[name] = torch.cat(
                                [weighted_single_weights[name], w],
                                dim=0
                            )
                            continue

                        continue

            else:
                # ====================================================
                # stacking WITHOUT zero padding
                # ====================================================
                if idx == 0:
                    weighted_single_weights = {}

                    for name, tensor in single_weights.items():

                        if is_lora_A(name):
                            # Freeze-A: use global
                            if freezeA_phase:
                                weighted_single_weights[name] = global_state[name]
                            else:
                                weighted_single_weights[name] = tensor
                            continue

                        if is_lora_B(name):
                            weighted_single_weights[name] = tensor * weights_array[idx]
                            continue

                        weighted_single_weights[name] = tensor

                else:
                    for name, tensor in single_weights.items():

                        if is_lora_A(name):
                            continue

                        if is_lora_B(name):
                            weighted_single_weights[name] = torch.cat(
                                [
                                    weighted_single_weights[name],
                                    tensor * weights_array[idx]
                                ],
                                dim=0
                            )
                            continue

                        continue

        # =======================================================
        # FedAvg (no stacking)
        # =======================================================
        else:
            if zero_padding:
                max_r = max(local_ranks)

                if idx == 0:
                    weighted_single_weights = {}
                    for name, tensor in single_weights.items():

                        if is_lora_A(name) and freezeA_phase:
                            w = global_state[name]
                        else:
                            w = tensor

                        if w.dim() == 2 and w.shape[0] < max_r and is_lora_B(name):
                            pad = ZeroPad2d((0, 0, 0, max_r - w.shape[0]))
                            w = pad(w)

                        weighted_single_weights[name] = w * weights_array[idx]

                else:
                    for name, tensor in single_weights.items():
                        w = tensor

                        if is_lora_A(name) and freezeA_phase:
                            w = global_state[name]

                        if w.dim() == 2 and w.shape[0] < max_r and is_lora_B(name):
                            pad = ZeroPad2d((0, 0, 0, max_r - w.shape[0]))
                            w = pad(w)

                        weighted_single_weights[name] += w * weights_array[idx]

            else:
                if idx == 0:
                    weighted_single_weights = {
                        name: w * weights_array[idx]
                        for name, w in single_weights.items()
                    }
                else:
                    for name, w in single_weights.items():
                        weighted_single_weights[name] += w * weights_array[idx]

    # ===========================================================
    # APPLY merged weights
    # ===========================================================
    if stacking:
        torch.save(
            weighted_single_weights,
            os.path.join(output_dir, str(epoch), "adapter_model.bin")
        )
        return model

    elif full:
        torch.save(
            weighted_single_weights,
            os.path.join(output_dir, str(epoch), "pytorch_model.bin")
        )
        model.load_state_dict(weighted_single_weights)
        return model

    else:
        set_peft_model_state_dict(model, weighted_single_weights, "default")
        return model
