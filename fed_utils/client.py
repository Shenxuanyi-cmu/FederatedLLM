# fed_utils/client.py

import transformers
import os
from datasets import load_dataset
import copy
from collections import OrderedDict
import torch
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from tqdm import tqdm


class GeneralClient:
    def __init__(self, client_id, model, data_path, output_dir, freezeA_phase=False):
        self.client_id = client_id
        self.model = model
        self.freezeA_phase = freezeA_phase

        # local dataset
        self.local_data_path = os.path.join(data_path, f"local_training_{client_id}.json")
        self.local_data = load_dataset("json", data_files=self.local_data_path)

        # output path
        self.output_dir = output_dir
        self.local_output_dir = os.path.join(
            output_dir, "trainer_saved", f"local_output_{client_id}"
        )

    # ===========================================================
    def preprare_local_dataset(self, generate_and_tokenize_prompt, local_val_set_size):
        ds = self.local_data["train"].shuffle()
        self.local_train_dataset = ds.map(generate_and_tokenize_prompt)
        self.local_eval_dataset = None
        self.local_val_set_size = local_val_set_size

    # ===========================================================
    def build_local_trainer(
        self,
        tokenizer,
        local_micro_batch_size,
        gradient_accumulation_steps,
        local_num_epochs,
        local_learning_rate,
        group_by_length,
        ddp,
    ):

        if self.freezeA_phase:
            # Freeze only A
            for name, p in self.model.named_parameters():
                if "lora_A" in name:
                    p.requires_grad = False

        self.train_args = transformers.TrainingArguments(
            per_device_train_batch_size=local_micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=local_num_epochs,
            learning_rate=local_learning_rate,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            eval_strategy="no",   # ✔ confirmed working on 4.46
            save_strategy="no",
            output_dir=self.local_output_dir,
            group_by_length=group_by_length,
            dataloader_drop_last=False,
        )

        self.local_trainer = transformers.Trainer(
            model=self.model,
            args=self.train_args,
            train_dataset=self.local_train_dataset,
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer,
                pad_to_multiple_of=8,
                return_tensors="pt",
                padding=True,
            ),
        )

    # ===========================================================
    def initiate_local_training(self):
        self.model.config.use_cache = False

        # backup old A/B
        self.params_dict_old = {
            n: p.detach().clone()
            for n, p in self.model.named_parameters()
            if "default" in n
        }

    # ===========================================================
    def train(self):
        self.local_trainer.train()

        # ===========================================================
        # ⭐ DEBUG PRINT A / B norm after local update
        # ===========================================================
        A_norm = None
        B_norm = None

        for name, p in self.model.named_parameters():
            if "lora_A" in name and A_norm is None:
                A_norm = p.norm().item()
            if "lora_B" in name and B_norm is None:
                B_norm = p.norm().item()
            if A_norm is not None and B_norm is not None:
                break

        print(
            f"[CLIENT {self.client_id}] "
            f"After Train → A_norm={A_norm:.6f}, B_norm={B_norm:.6f}"
        )

    # ===========================================================
    def terminate_local_training(self, epoch, local_dataset_len_dict, previously_selected):

        local_dataset_len_dict[self.client_id] = len(self.local_train_dataset)

        # obtain trained adapter state
        trained = get_peft_model_state_dict(self.model, adapter_name="default")

        # upload dict
        upload_dict = {}
        for name, tensor in trained.items():
            if self.freezeA_phase:
                if "lora_B" in name:   # freeze phase only upload B
                    upload_dict[name] = tensor.cpu()
            else:
                upload_dict[name] = tensor.cpu()

        # save to disk
        save_dir = os.path.join(self.output_dir, str(epoch), f"local_output_{self.client_id}")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(upload_dict, os.path.join(save_dir, "pytorch_model.bin"))

        # restore original A/B
        set_peft_model_state_dict(self.model, self.params_dict_old, "default")

        previously_selected |= {self.client_id}
        return self.model, local_dataset_len_dict, previously_selected
