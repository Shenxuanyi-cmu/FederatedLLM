# ===========================================================
# final_main.py  ——  FedIT + Freeze-A only + norm check
# ===========================================================

import os
import fire
import torch
import copy
import glob
from tqdm.auto import tqdm
import wandb

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

import datasets
from utils.prompter import Prompter
from fed_utils.client import GeneralClient
from fed_utils.model_aggregation import FedAvg
from fed_utils.evaluation import global_evaluation


datasets.tqdm = lambda *args, **kwargs: tqdm(*args, **kwargs, leave=False)
HF_TOKEN = "hf_xxxxx"


# ===========================================================
# Freeze A helper
# ===========================================================
def freeze_lora_A(model):
    for n, p in model.named_parameters():
        if "lora_A" in n:
            p.requires_grad = False


# ===========================================================
# Main Federated Finetuning
# ===========================================================
def fl_finetune(
    global_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    data_path="./data_wiz",
    output_dir="./runs/FedIT/",

    num_communication_rounds=3,
    num_clients=2,

    local_batch_size=128,
    local_micro_batch_size=16,
    local_num_epochs=1,
    local_learning_rate=3e-4,
    local_val_set_size=0,
    cutoff_len=256,

    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_target_modules=("q_proj", "v_proj"),

    freezeA_after_rounds=-1,
):

    print(f"🚀 FedIT Training Started. Freeze-A from round ≥ {freezeA_after_rounds}")

    wandb.init(
        project="FedIT-FL",
        name=f"fedIT_clients{num_clients}_rounds{num_communication_rounds}",
    )

    # auto-select numeric subdir
    subdirs = [d for d in os.listdir(data_path) if d.isdigit()]
    if subdirs:
        data_path = os.path.join(data_path, max(subdirs, key=int))

    # count clients
    all_json = sorted(glob.glob(os.path.join(data_path, "local_training_*.json")))
    num_clients = min(num_clients, len(all_json))
    print(f"📦 Using {num_clients} clients")

    # -----------------------------------------------------------
    # Load global model
    # -----------------------------------------------------------
    prompter = Prompter("alpaca")
    quant = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        global_model,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quant,
        token=HF_TOKEN,
    )

    tokenizer = AutoTokenizer.from_pretrained(global_model, token=HF_TOKEN)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    def tokenize_prompt(prompt):
        r = tokenizer(prompt, truncation=True, max_length=cutoff_len, padding=False)
        r["labels"] = r["input_ids"].copy()
        return r

    def generate_and_tokenize(dp):
        p = prompter.generate_prompt(dp["instruction"], None, dp["output"])
        return tokenize_prompt(p)

    model = prepare_model_for_kbit_training(model)

    # enable lora
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


    print("\n====================")
    print("🔥 Baseline evaluation on raw model (no training)")
    print("====================")

    baseline_acc = global_evaluation(model, tokenizer, prompter, "./mmlu_test_1444.jsonl")
    print(f"🌟 Baseline Acc: {baseline_acc}")
    wandb.log({"baseline_accuracy": baseline_acc})


    # output dir
    output_dir = os.path.join(output_dir, str(num_clients))
    os.makedirs(output_dir, exist_ok=True)

    local_dataset_len_dict = {}
    previously_selected_clients_set = set()

    # ===========================================================
    # Federated Rounds
    # ===========================================================
    for epoch in range(num_communication_rounds):

        print(f"\n🔥 ROUND {epoch}")

        # --------------------------------
        # DEBUG: print LoRA_B at round start
        # --------------------------------
        try:
            key_A = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
            key_B = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight"

            wA = model.state_dict()[key_A]
            wB = model.state_dict()[key_B]

            print(f"[DEBUG] Round {epoch} START A_norm = {wA.norm().item():.6f}")
            print(f"[DEBUG] Round {epoch} START B_norm = {wB.norm().item():.6f}")

        except Exception as e:
            print("[DEBUG] Cannot read A/B at round start:", e)


        freezeA_phase = (freezeA_after_rounds >= 0 and epoch >= freezeA_after_rounds)
        if freezeA_phase:
            print("🔒 Freeze-A Phase")

        selected_clients = list(range(num_clients))

        # -----------------------------
        # Local training
        # -----------------------------
        for cid in selected_clients:

            model_client = copy.deepcopy(model)

            if freezeA_phase:
                freeze_lora_A(model_client)

            client = GeneralClient(
                cid, model_client, data_path, output_dir, freezeA_phase=freezeA_phase
            )

            client.preprare_local_dataset(generate_and_tokenize, local_val_set_size)
            client.build_local_trainer(
                tokenizer,
                local_micro_batch_size,
                local_batch_size // local_micro_batch_size,
                local_num_epochs,
                local_learning_rate,
                False,
                False,
            )

            client.initiate_local_training()
            client.train()
            model_client, local_dataset_len_dict, previously_selected_clients_set = \
                client.terminate_local_training(
                    epoch, local_dataset_len_dict, previously_selected_clients_set
                )

        # -----------------------------
        # server aggregation
        # -----------------------------
        print("📦 Aggregating (FedIT)...")
        model = FedAvg(
            model,
            selected_clients,
            output_dir,
            local_dataset_len_dict,
            epoch,
            freezeA_phase=freezeA_phase,
        )

        # --------------------------------
        # DEBUG: print LoRA_B after FedAvg
        # --------------------------------
        try:
            wA = model.state_dict()[key_A]
            wB = model.state_dict()[key_B]
            print(f"[DEBUG] After FedAvg A_norm = {wA.norm().item():.6f}")
            print(f"[DEBUG] After FedAvg B_norm = {wB.norm().item():.6f}")
        except:
            print("[DEBUG] Cannot read A/B after FedAvg")


        # global eval
        acc = global_evaluation(model, tokenizer, prompter, "./mmlu_test_1444.jsonl")
        print(f"🌟 Acc of Round {epoch}: {acc}")
        wandb.log({"round": epoch, "accuracy": acc})

    print("🎉 FedIT Training Completed!")


if __name__ == "__main__":
    fire.Fire(fl_finetune)

