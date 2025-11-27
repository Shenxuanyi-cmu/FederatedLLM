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
        self.local_output_dir = os.path.join(output_dir, "trainer_saved", f"local_output_{client_id}")

    # ===========================================================
    # dataset prepare
    # ===========================================================
    def preprare_local_dataset(self, generate_and_tokenize_prompt, local_val_set_size, use_tqdm=True):

        def map_with_tqdm(ds, desc):
            total = len(ds)
            pbar = tqdm(total=total, desc=desc, disable=not use_tqdm)

            def wrapper(x):
                pbar.update(1)
                return generate_and_tokenize_prompt(x)

            mapped = ds.map(wrapper)
            pbar.close()
            return mapped

        ds = self.local_data["train"].shuffle()

        if local_val_set_size > 0:
            split = ds.train_test_split(test_size=local_val_set_size, seed=42)
            self.local_train_dataset = map_with_tqdm(split["train"], f"Client {self.client_id} tokenize(train)")
            self.local_eval_dataset = map_with_tqdm(split["test"], f"Client {self.client_id} tokenize(eval)")
        else:
            self.local_train_dataset = map_with_tqdm(ds, f"Client {self.client_id} tokenize")
            self.local_eval_dataset = None

        self.local_val_set_size = local_val_set_size

    # ===========================================================
    # Trainer build
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
            for name, p in self.model.named_parameters():
                if "lora_A" in name:
                    p.requires_grad = False

        # ❗❗ transformers>=4.46 MUST use evaluation_strategy
        self.train_args = transformers.TrainingArguments(
            per_device_train_batch_size=local_micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=local_num_epochs,
            learning_rate=local_learning_rate,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",

            # ❌ 禁用 evaluation（FL 不需要）
            eval_strategy="no",

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
    # Local training init
    # ===========================================================
    def initiate_local_training(self):
        self.model.config.use_cache = False

        # old LoRA weights backup
        self.params_dict_old = copy.deepcopy(
            OrderedDict((n, p.detach()) for n, p in self.model.named_parameters() if "default" in n)
        )

        # new LoRA to train
        self.params_dict_new = OrderedDict(
            (n, p.detach()) for n, p in self.model.named_parameters() if "default" in n
        )

        # monkey patch
        self.model.state_dict = (
            lambda inst, *_, **__: get_peft_model_state_dict(inst, self.params_dict_new, "default")
        ).__get__(self.model, type(self.model))

    # ===========================================================
    # train
    # ===========================================================
    def train(self):
        self.local_trainer.train()

    # ===========================================================
    # terminate
    # ===========================================================
    def terminate_local_training(self, epoch, local_dataset_len_dict, previously_selected_clients_set):

        local_dataset_len_dict[self.client_id] = len(self.local_train_dataset)

        # trained LoRA
        new_lora = self.model.state_dict()

        save_dir = os.path.join(self.output_dir, str(epoch), f"local_output_{self.client_id}")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(new_lora, os.path.join(save_dir, "pytorch_model.bin"))

        # restore old LoRA
        old_lora = get_peft_model_state_dict(self.model, self.params_dict_old, "default")
        set_peft_model_state_dict(self.model, old_lora, "default")

        previously_selected_clients_set = previously_selected_clients_set | {self.client_id}

        return self.model, local_dataset_len_dict, previously_selected_clients_set

