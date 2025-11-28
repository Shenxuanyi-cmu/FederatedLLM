# main.py
import os
from typing import List
import fire
import torch
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
import copy
import glob

from utils.prompter import Prompter
from fed_utils.client import GeneralClient
from fed_utils.model_aggregation import FedAvg
from fed_utils.evaluation import global_evaluation


datasets.tqdm = lambda *args, **kwargs: tqdm(*args, **kwargs, leave=False)
HF_TOKEN = "hf_xxxxx"


# ===========================================================
# freeze A
# ===========================================================
def freeze_lora_A(model):
    for n, p in model.named_parameters():
        if "lora_A" in n:
            p.requires_grad = False


# ===========================================================
# main FL finetune
# ===========================================================
def fl_finetune(
    global_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    data_path="./data_wiz",
    output_dir="./runs/FLoRA-modern/",

    client_selection_frac=1.0,
    num_communication_rounds=3,
    num_clients=2,

    local_batch_size=128,
    local_micro_batch_size=16,
    local_num_epochs=1,
    local_learning_rate=3e-4,
    local_val_set_size=0,
    cutoff_len=512,

    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_target_modules=("q_proj", "v_proj"),

    stacking=False,
    heter=False,
    local_ranks=(),
    zero_padding=False,
    full=False,

    freezeA_after_rounds=-1,
):

    print(f"🚀 FL Training Started. Freeze-A from round ≥ {freezeA_after_rounds}")
    wandb.init(
        project="FLoRA-FL",
        name=f"run_clients{num_clients}_rounds{num_communication_rounds}",
        config={
            "clients": num_clients,
            "rounds": num_communication_rounds,
            "stacking": stacking,
            "freezeA_after_rounds": freezeA_after_rounds,
            "lora_r": lora_r,
            "cutoff_len": cutoff_len,
        }
    )

    # auto-select 10/20
    subdirs = [d for d in os.listdir(data_path) if d.isdigit()]
    if subdirs:
        data_path = os.path.join(data_path, max(subdirs, key=int))

    all_json = sorted(glob.glob(os.path.join(data_path, "local_training_*.json")))
    num_clients = min(num_clients, len(all_json))
    print(f"📦 Using {num_clients} clients")

    # -----------------------------
    # load model
    # -----------------------------
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

    def tokenize(prompt):
        r = tokenizer(prompt, truncation=True, max_length=cutoff_len, padding=False)
        r["labels"] = r["input_ids"].copy()
        return r

    def generate_and_tokenize_prompt(dp):
        p = prompter.generate_prompt(dp["instruction"], None, dp["output"])
        return tokenize(p)

    model = prepare_model_for_kbit_training(model)

    if not stacking and not full:
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

    # -----------------------------
    # prepare FL
    # -----------------------------
    output_dir = os.path.join(output_dir, str(num_clients))
    os.makedirs(output_dir, exist_ok=True)

    local_dataset_len_dict = {}
    previously_selected_clients_set = set()

    # ===========================================================
    # FL rounds
    # ===========================================================
    for epoch in range(num_communication_rounds):
        print(f"\n🔥 ROUND {epoch}")

        freezeA_phase = (freezeA_after_rounds >= 0 and epoch >= freezeA_after_rounds)
        if freezeA_phase:
            print("🔒 Freeze-A Phase")

        selected_clients = list(range(num_clients))

        # -----------------------------
        # local training
        # -----------------------------
        for cid in selected_clients:

            model_client = copy.deepcopy(model)

            if stacking:
                cfg = LoraConfig(
                    base_model_name_or_path=global_model,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=list(lora_target_modules),
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                model_client = get_peft_model(model_client, cfg)

            if freezeA_phase:
                freeze_lora_A(model_client)

            client = GeneralClient(cid, model_client, data_path, output_dir, freezeA_phase)

            client.preprare_local_dataset(generate_and_tokenize_prompt, local_val_set_size)
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
                    epoch,
                    local_dataset_len_dict,
                    previously_selected_clients_set
                )

        # -----------------------------
        # aggregation
        # -----------------------------
        print("📦 Aggregating...")
        model = FedAvg(
            model,
            selected_clients,
            output_dir,
            local_dataset_len_dict,
            epoch,
            stacking,
            lora_r,
            heter,
            local_ranks,
            zero_padding,
            full,
            freezeA_phase=freezeA_phase,
        )

        # global eval
        acc = global_evaluation(model, tokenizer, prompter, "./mmlu_test_1444.jsonl")
        print(f"🌟 Acc of Round {epoch}: {acc}")
        wandb.log({"round": epoch, "accuracy": acc})


    print("🎉 FL Training Completed!")


if __name__ == "__main__":
    fire.Fire(fl_finetune)
