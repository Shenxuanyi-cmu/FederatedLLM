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

        self.local_data_path = os.path.join(data_path, f"local_training_{self.client_id}.json")
        self.local_data = load_dataset("json", data_files=self.local_data_path)

        self.output_dir = output_dir
        self.local_output_dir = os.path.join(output_dir, "trainer_saved", f"local_output_{self.client_id}")

    # ===========================================================
    # dataset
    # ===========================================================
    def preprare_local_dataset(self, generate_and_tokenize_prompt, local_val_set_size, use_tqdm=True):

        def map_with_tqdm(dataset, desc):
            total = len(dataset)
            pbar = tqdm(total=total, desc=desc, disable=not use_tqdm)

            def wrapper(example):
                pbar.update(1)
                return generate_and_tokenize_prompt(example)

            mapped = dataset.map(wrapper)
            pbar.close()
            return mapped

        if local_val_set_size > 0:
            split = self.local_data["train"].train_test_split(
                test_size=local_val_set_size, shuffle=True, seed=42
            )

            self.local_train_dataset = map_with_tqdm(
                split["train"].shuffle(),
                desc=f"Client {self.client_id} tokenize(train)"
            )

            self.local_eval_dataset = map_with_tqdm(
                split["test"].shuffle(),
                desc=f"Client {self.client_id} tokenize(eval)"
            )

        else:
            self.local_train_dataset = map_with_tqdm(
                self.local_data["train"].shuffle(),
                desc=f"Client {self.client_id} tokenize"
            )
            self.local_eval_dataset = None

        self.local_val_set_size = local_val_set_size

    # ===========================================================
    # Trainer
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
            for name, param in self.model.named_parameters():
                if "lora_A" in name:
                    param.requires_grad = False

        self.train_args = transformers.TrainingArguments(
            per_device_train_batch_size=local_micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
            num_train_epochs=local_num_epochs,
            learning_rate=local_learning_rate,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            eval_strategy="steps" if self.local_val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if self.local_val_set_size > 0 else None,
            save_steps=5000000,
            output_dir=self.local_output_dir,
            save_total_limit=1,
            load_best_model_at_end=True if self.local_val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            dataloader_drop_last=False,
        )

        self.local_trainer = transformers.Trainer(
            model=self.model,
            train_dataset=self.local_train_dataset,
            eval_dataset=self.local_eval_dataset,
            args=self.train_args,
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )

    # -----------------------------------------------------------
    def initiate_local_training(self):
        self.model.config.use_cache = False

        self.params_dict_old = copy.deepcopy(
            OrderedDict(
                (name, p.detach())
                for name, p in self.model.named_parameters()
                if "default" in name
            )
        )

        self.params_dict_new = OrderedDict(
            (name, p.detach())
            for name, p in self.model.named_parameters()
            if "default" in name
        )

        self.model.state_dict = (
            lambda instance, *_, **__: get_peft_model_state_dict(
                instance, self.params_dict_new, "default"
            )
        ).__get__(self.model, type(self.model))

    # -----------------------------------------------------------
    def train(self):
        self.local_trainer.train()

    # -----------------------------------------------------------
    def terminate_local_training(self, epoch, local_dataset_len_dict, previously_selected_clients_set):

        local_dataset_len_dict[self.client_id] = len(self.local_train_dataset)

        new_adapter_weight = self.model.state_dict()

        save_dir = os.path.join(self.output_dir, str(epoch), f"local_output_{self.client_id}")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(new_adapter_weight, os.path.join(save_dir, "pytorch_model.bin"))

        old_weight = get_peft_model_state_dict(self.model, self.params_dict_old, "default")
        set_peft_model_state_dict(self.model, old_weight, "default")

        previously_selected_clients_set = previously_selected_clients_set | {self.client_id}
        return self.model, local_dataset_len_dict, previously_selected_clients_set, self.client_id
