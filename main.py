import os
from typing import List
from tqdm import tqdm
import fire
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# 🔧 peft
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)

# federated utils
from fed_utils import FedAvg, client_selection, global_evaluation, GeneralClient
from utils.prompter import Prompter

import glob
import copy

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
if HF_TOKEN is None:
    print("⚠️ Warning: HF_TOKEN not found in environment variables.")


# ===========================================================
# Freeze-A helpers
# ===========================================================
def freeze_lora_A(model):
    """Freeze all lora_A parameters so only B is trainable."""
    for name, param in model.named_parameters():
        if "lora_A" in name:
            param.requires_grad = False


# ===========================================================
# Main FL logic
# ===========================================================
def fl_finetune(
    # basic params
    global_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    data_path: str = "./data_wiz",
    output_dir: str = "./runs/FLoRA-modern/",

    # FL settings
    client_selection_strategy: str = "random",
    client_selection_frac: float = 1,
    num_communication_rounds: int = 3,
    num_clients: int = 10,

    # local training params
    local_batch_size: int = 128,
    local_micro_batch_size: int = 16,
    local_num_epochs: int = 1,
    local_learning_rate: float = 3e-4,
    local_val_set_size: int = 0,
    cutoff_len: int = 512,

    # LoRA params
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ("q_proj", "v_proj"),

    # llm settings
    group_by_length: bool = False,
    prompt_template_name: str = "alpaca",

    # aggregation mode
    stacking: bool = False,

    # eval
    dev_data_path: str = "./mmlu_test_1444.jsonl",

    # heterogeneous
    heter: bool = False,
    local_ranks: List[int] = (64, 32, 16, 16, 8, 8, 4, 4, 4, 4),
    zero_padding: bool = False,
    full: bool = False,

    # ⭐ NEW: freeze A after N rounds
    freezeA_after_rounds: int = -1,
):

    num_clients = int(num_clients)
    num_communication_rounds = int(num_communication_rounds)
    freezeA_after_rounds = int(freezeA_after_rounds)

    print(f"🚀 Starting FL + LoRA, Freeze-A at round {freezeA_after_rounds}")

    # ===========================================================
    # Auto-select dataset folder
    # ===========================================================
    subdirs = [d for d in os.listdir(data_path) if d.isdigit()]
    if subdirs:
        max_subdir = max(subdirs, key=int)
        data_path = os.path.join(data_path, max_subdir)

    assert os.path.exists(data_path), f"❌ Missing data folder {data_path}"

    all_client_files = sorted(glob.glob(os.path.join(data_path, "local_training_*.json")))
    num_clients = min(num_clients, len(all_client_files))
    print(f"📦 Using {num_clients} clients")

    # ===========================================================
    # Model init
    # ===========================================================
    prompter = Prompter(prompt_template_name)
    gradient_accumulation_steps = local_batch_size // local_micro_batch_size

    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        global_model,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quant_config,
        token=HF_TOKEN,
    )

    tokenizer = AutoTokenizer.from_pretrained(global_model, token=HF_TOKEN)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        r = tokenizer(prompt, truncation=True, max_length=cutoff_len,
                       padding=False, return_tensors=None)
        if add_eos_token and r["input_ids"][-1] != tokenizer.eos_token_id:
            if len(r["input_ids"]) < cutoff_len:
                r["input_ids"].append(tokenizer.eos_token_id)
                r["attention_mask"].append(1)
        r["labels"] = r["input_ids"].copy()
        return r

    def generate_and_tokenize_prompt(dp):
        if "context" in dp:
            prompt_text = prompter.generate_prompt(
                dp.get("instruction",""), dp.get("context",""), dp.get("output","")
            )
        else:
            prompt_text = prompter.generate_prompt(
                dp.get("instruction",""), dp.get("input",""), dp.get("output","")
            )
        return tokenize(prompt_text)

    # LoRA
    model = prepare_model_for_kbit_training(model)
    if not full and not stacking:
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
    # Output folder
    # ===========================================================
    output_dir = os.path.join(output_dir, str(num_clients))
    os.makedirs(output_dir, exist_ok=True)

    # ===========================================================
    # FL rounds
    # ===========================================================
    acc_list = []
    local_dataset_len_dict = {}
    previously_selected_clients_set = set()
    last_client_id = None

    for epoch in tqdm(range(num_communication_rounds)):
        print(f"\n🔥 Round {epoch}")

        # ⭐ 判断当前是否在冻结 A 的阶段
        freezeA_phase = (freezeA_after_rounds >= 0 and epoch >= freezeA_after_rounds)
        if freezeA_phase:
            print("🔒 Freeze-A phase ON (only training B)")

        selected_clients_set = client_selection(
            num_clients, client_selection_frac, client_selection_strategy, other_info=epoch
        )

        # ================== local training ==================
        for cid in selected_clients_set:
            model_client = copy.deepcopy(model)

            # ⭐ 客户端冻结 A
            if freezeA_phase:
                freeze_lora_A(model_client)

            client = GeneralClient(cid, model_client, data_path, output_dir)
            client.preprare_local_dataset(generate_and_tokenize_prompt, local_val_set_size)
            client.build_local_trainer(
                tokenizer,
                local_micro_batch_size,
                gradient_accumulation_steps,
                local_num_epochs,
                local_learning_rate,
                group_by_length,
                False,
            )
            client.initiate_local_training()
            client.train()
            (
                model_client, 
                local_dataset_len_dict,
                previously_selected_clients_set,
                last_client_id
            ) = client.terminate_local_training(
                epoch, local_dataset_len_dict, previously_selected_clients_set
            )
            del client

        # ================== aggregation ==================
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

            # ⭐ 新增：让 FedAvg 只聚合 B
            freezeA_phase=freezeA_phase,
        )

        # ================== eval ==================
        acc = global_evaluation(model, tokenizer, prompter, dev_data_path)
        acc_list.append(acc)
        print(f"📊 Acc epoch {epoch}: {acc}")

    # ================== save log ==================
    with open(os.path.join(output_dir, "log.txt"), "a") as f:
        for a in acc_list:
            f.write(str(a) + "\n")

    print("🎉 FL training finished!")


if __name__ == "__main__":
    fire.Fire(fl_finetune)
