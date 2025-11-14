import os
from typing import List
from tqdm import tqdm
import fire
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
)

# 🔧 peft
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    AdaLoraConfig,
)

# federated utils
from fed_utils import FedAvg, client_selection, global_evaluation, GeneralClient
from utils.prompter import Prompter

import numpy as np
import random
import copy
import glob

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
if HF_TOKEN is None:
    print("⚠️ Warning: HF_TOKEN not found in environment variables.")


# ===========================================================
# NEW: helpers for Freeze-A mechanism
# ===========================================================
def freeze_lora_A(model):
    """Freeze all lora_A parameters so only B is trainable."""
    for name, param in model.named_parameters():
        if "lora_A" in name:
            param.requires_grad = False


def extract_and_save_A(model, save_path):
    """Extract all lora_A parameters and save them."""
    state_dict = model.state_dict()
    A_only = {}
    for name, param in state_dict.items():
        if "lora_A" in name:
            A_only[name] = param.cpu()
    torch.save(A_only, save_path)
    print(f"✅ Saved A_global to {save_path}")


# ===========================================================
# main FL process
# ===========================================================
def fl_finetune(
    # model/data params
    global_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    data_path: str = "./data_wiz",
    output_dir: str = "./runs/FLoRA-modern/",
    # FL hyperparams
    client_selection_strategy: str = "random",
    client_selection_frac: float = 1,
    num_communication_rounds: int = 3,
    num_clients: int = 10,
    # Local training hyperparams
    local_batch_size: int = 128,
    local_micro_batch_size: int = 16,
    local_num_epochs: int = 1,
    local_learning_rate: float = 3e-4,
    local_val_set_size: int = 0,
    local_save_steps: int = 3,
    cutoff_len: int = 512,
    # LoRA hyperparams
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ("q_proj", "v_proj"),
    # llm hyperparams
    train_on_inputs: bool = True,
    group_by_length: bool = False,
    resume_from_checkpoint: str = None,
    prompt_template_name: str = "alpaca",
    # aggregation mode
    stacking: bool = False,
    # evaluation
    dev_data_path: str = "./mmlu_test_1444.jsonl",
    # heterogeneous
    heter: bool = False,
    local_ranks: List[int] = (64, 32, 16, 16, 8, 8, 4, 4, 4, 4),
    zero_padding: bool = False,
    Adalora: bool = False,
    full: bool = False,

    # ====================================================
    # NEW: two-phase mode
    # ====================================================
    mode: str = "full_sa",   # ["full_sa", "freeze_A"]
    A_path: str = "",        # freeze_A 模式下加载 A_global 的路径
):
    
    # ------ force convert command-line Fire inputs ------
    num_clients = int(num_clients)
    num_communication_rounds = int(num_communication_rounds)
    client_selection_frac = float(client_selection_frac)
    local_batch_size = int(local_batch_size)
    local_micro_batch_size = int(local_micro_batch_size)
    local_num_epochs = int(local_num_epochs)
    local_learning_rate = float(local_learning_rate)
    local_val_set_size = int(local_val_set_size)

    # ===========================================================
    # Print basic info
    # ===========================================================
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"🚀 FL + LoRA starting, mode={mode}")

    assert global_model, "Please provide --global_model"

    # ===========================================================
    # Data auto select
    # ===========================================================
    subdirs = [d for d in os.listdir(data_path) if d.isdigit()]
    if subdirs:
        max_subdir = max(subdirs, key=int)
        candidate_path = os.path.join(data_path, max_subdir)
        if os.path.exists(candidate_path):
            data_path = candidate_path
            print(f"📂 Auto-selected data folder: {data_path}")
        else:
            data_path = os.path.join(data_path, str(num_clients))

    assert os.path.exists(data_path), f"❌ Data folder missing: {data_path}"

    # pick first num_clients files
    all_client_files = sorted(glob.glob(os.path.join(data_path, "local_training_*.json")))
    available_clients = len(all_client_files)
    assert available_clients > 0
    num_clients = min(num_clients, available_clients)
    print(f"📦 Using {num_clients} clients")

    # ===========================================================
    # Model + tokenizer
    # ===========================================================
    gradient_accumulation_steps = local_batch_size // local_micro_batch_size
    prompter = Prompter(prompt_template_name)
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = max(1, gradient_accumulation_steps // world_size)

    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        global_model,
        device_map=device_map,
        torch_dtype=torch.float16,
        quantization_config=quant_config,
        token=HF_TOKEN,
    )

    tokenizer = AutoTokenizer.from_pretrained(global_model, token=HF_TOKEN)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # tokenizer wrapper
    def tokenize(prompt, add_eos_token=True):
        r = tokenizer(prompt, truncation=True, max_length=cutoff_len,
                       padding=False, return_tensors=None)
        if r["input_ids"][-1] != tokenizer.eos_token_id and len(r["input_ids"]) < cutoff_len:
            if add_eos_token:
                r["input_ids"].append(tokenizer.eos_token_id)
                r["attention_mask"].append(1)
        r["labels"] = r["input_ids"].copy()
        return r

    def generate_and_tokenize_prompt(dp):
        if "context" in dp:
            full = prompter.generate_prompt(
                dp.get("instruction", ""),
                dp.get("context", ""),
                dp.get("response", dp.get("output", "")),
            )
        else:
            full = prompter.generate_prompt(
                dp.get("instruction", ""),
                dp.get("input", ""),
                dp.get("output", ""),
            )
        tok = tokenize(full)
        return tok

    # prepare 8bit training
    model = prepare_model_for_kbit_training(model)

    # ===========================================================
    # LoRA init
    # ===========================================================
    if not full:
        if not stacking:
            if zero_padding:
                config_ori = LoraConfig(
                    base_model_name_or_path=global_model,
                    r=max(local_ranks),
                    lora_alpha=lora_alpha * max(local_ranks),
                    target_modules=list(lora_target_modules),
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
            else:
                config = LoraConfig(
                    base_model_name_or_path=global_model,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=list(lora_target_modules),
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, config)

    # ===========================================================
    # NEW: Freeze-A mode → load A and freeze
    # ===========================================================
    if mode == "freeze_A":
        assert A_path != "", "freeze_A 模式必须提供 --A_path"
        print(f"🔒 Loading A_global from {A_path}")
        A_only = torch.load(A_path, map_location="cpu")

        missing, unexpected = model.load_state_dict(A_only, strict=False)
        print("Loaded A. Missing:", missing, "Unexpected:", unexpected)

        # freeze all A
        freeze_lora_A(model)
        print("🔒 All lora_A parameters frozen.")

    # parallel
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    output_dir = os.path.join(output_dir, str(num_clients))
    os.makedirs(output_dir, exist_ok=True)

    acc_list = []
    local_dataset_len_dict = {}
    previously_selected_clients_set = set()
    last_client_id = None

    print("🚀 Starting federated rounds...")

    # ===========================================================
    # FL rounds
    # ===========================================================
    for epoch in tqdm(range(num_communication_rounds)):

        selected_clients_set = client_selection(
            num_clients,
            client_selection_frac,
            client_selection_strategy,
            other_info=epoch,
        )

        for client_id in selected_clients_set:

            # clone global model to client
            model_client = copy.deepcopy(model)

            # NEW: ensure A is frozen in client too
            if mode == "freeze_A":
                freeze_lora_A(model_client)

            # build client
            client = GeneralClient(client_id, model_client, data_path, output_dir)

            client.preprare_local_dataset(generate_and_tokenize_prompt, local_val_set_size)
            client.build_local_trainer(
                tokenizer,
                local_micro_batch_size,
                gradient_accumulation_steps,
                local_num_epochs,
                local_learning_rate,
                group_by_length,
                ddp,
            )

            client.initiate_local_training()
            client.train()

            (model_client,
             local_dataset_len_dict,
             previously_selected_clients_set,
             last_client_id) = client.terminate_local_training(
                epoch, local_dataset_len_dict, previously_selected_clients_set
            )

            del client

        # ===========================================================
        # aggregation
        # ===========================================================
        print("📦 FedAvg aggregation...")
        model = FedAvg(
            model,
            selected_clients_set,
            output_dir,
            local_dataset_len_dict,
            epoch,
            stacking,
            lora_r,
            heter,
            list(local_ranks),
            zero_padding,
            full,
        )

        # save current round
        save_dir = os.path.join(output_dir, str(epoch))
        os.makedirs(save_dir, exist_ok=True)

        if not full:
            if stacking:
                config_ori.save_pretrained(save_dir)
                model = PeftModel.from_pretrained(model, save_dir)
            else:
                torch.save(model.state_dict(), os.path.join(save_dir, "adapter_model.bin"))
                if "config" in locals():
                    config.save_pretrained(save_dir)

        # ===========================================================
        # global evaluation
        # ===========================================================
        acc = global_evaluation(model, tokenizer, prompter, dev_data_path)
        acc_list.append(acc)
        print(f"📊 Acc epoch {epoch}: {acc}")

        # delete history
        if epoch < num_communication_rounds - 1:
            os.system(f"rm -rf {save_dir}")

    # ===========================================================
    # NEW: If warmup, save A_global
    # ===========================================================
    if mode == "full_sa":
        A_save_path = os.path.join(output_dir, "A_global.pt")
        extract_and_save_A(model, A_save_path)

    # log
    filename = output_dir + "/log.txt"
    with open(filename, "a") as f:
        for a in acc_list:
            f.write(str(a) + "\n")
    print("✅ Log saved")


if __name__ == "__main__":
    fire.Fire(fl_finetune)
