from peft import set_peft_model_state_dict
import torch
import os
from torch.nn.functional import normalize

def is_lora_A(name):
    return ("lora_A" in name) and ("default" in name)

def is_lora_B(name):
    return ("lora_B" in name) and ("default" in name)

# ===========================================================
#          FedIT + Freeze-A (最终最简洁，可运行)
# ===========================================================
def FedAvg(
    model,
    selected_clients,
    output_dir,
    local_dataset_len_dict,
    epoch,
    freezeA_phase=False
):
    """FedAvg 版本：只平均 A 和 B；freeze-A 时只平均 B"""

    # FedAvg 权重（按数据量）
    weights = normalize(
        torch.tensor([local_dataset_len_dict[c] for c in selected_clients], dtype=torch.float32),
        p=1, dim=0
    )

    # 当前 global A (freeze 时需要保留)
    global_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    # 初始化 acc
    merged = None

    # =====================================================
    #            聚合 LoRA 权重
    # =====================================================
    for idx, cid in enumerate(selected_clients):

        ckpt = os.path.join(output_dir, str(epoch),
                            f"local_output_{cid}", "pytorch_model.bin")
        local_state = torch.load(ckpt, map_location="cpu")

        # 处理每一个 local client
        if idx == 0:
            # 初始化 merged
            merged = {k: torch.zeros_like(v) for k, v in local_state.items()}

        for name, w in local_state.items():

            # freeze-A: 不用 local A_i，直接保留全局 A
            if freezeA_phase and is_lora_A(name):
                merged[name] = global_state[name]
                continue

            # B 和未冻结的 A 都做 FedAvg
            merged[name] += w * weights[idx]

    # =====================================================
    #         写入 global model 作为下一轮的 weights
    # =====================================================
    set_peft_model_state_dict(model, merged, "default")
    return model
