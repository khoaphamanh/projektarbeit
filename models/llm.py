import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
import transformers
from peft import LoraConfig
from trl import SFTTrainer
import sys
import os
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torch.utils.data import DataLoader
from timeit import default_timer
import random
import numpy as np
import re

# import neptune

# import preprocessing file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.preprocessing import DataPreprocessing


class LLM(DataPreprocessing):
    def __init__(self, name_data, seed):
        super().__init__(name_data, seed)

        # name of the model
        self.name_model_llama_2 = "meta-llama/Llama-2-7b-chat-hf"

        # path of the files and directories
        self.path_models_directory = os.path.dirname(os.path.abspath(__file__))

        # device
        self.device_map = "auto"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.gpu_name = torch.cuda.get_device_name(0)
            self.n_gpus = torch.cuda.device_count()
            self.vram = (
                torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
            )
        else:
            self.gpu_name = None
            self.n_gpus = None
            self.vram = None

    def load_model(self, quantized=False):
        """
        load model llama-2 if QLoRA or only use LoRA
        """
        # check if use Q-LoRA
        if quantized:
            use_4bit = True
            bnb_4bit_quant_type = "nf4"
            compute_dtype = torch.float16
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=use_4bit,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
            )

            model = AutoModelForCausalLM.from_pretrained(
                self.name_model_llama_2,
                quantization_config=bnb_config,
                device_map=self.device_map,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.name_model_llama_2,
                device_map=self.device_map,
            )

        model.config.use_cache = False
        model.config.pretraining_tp = 1

        return model

    def load_tokenizer(self):
        """
        load tokenizer for llama-2 model
        """
        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.name_model_llama_2)

        # Set pad_token to eos_token
        tokenizer.pad_token = tokenizer.eos_token

        # Set padding direction
        tokenizer.padding_side = "right"

        return tokenizer

    def load_peft_config(self, lora_r, lora_alpha, lora_dropout):
        """
        load LoRA Config
        """
        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CAUSAL_LM",
        )

        return peft_config

    def load_data(
        self,
        extracted_label=None,
        normalize=False,
        downsampling_n_instances=None,
        downsampling_n_instances_train=None,
        downsampling_n_instances_test=None,
        name_feature=False,
        save=False,
    ):
        """
        load data in llama-2 format
        """
        train_datasets, test_datasets = self.load_data_llm_format(
            extracted_label=extracted_label,
            normalize=normalize,
            downsampling_n_instances=downsampling_n_instances,
            downsampling_n_instances_train=downsampling_n_instances_train,
            downsampling_n_instances_test=downsampling_n_instances_test,
            name_feature=name_feature,
            save=save,
        )

        return train_datasets, test_datasets

    def classification(
        self,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        extracted_label=None,
        normalize=False,
        downsampling_n_instances=None,
        downsampling_n_instances_train=None,
        downsampling_n_instances_test=None,
        name_feature=False,
        save=False,
        quantized=False,
        batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=0.0001,
        weight_decay=0.001,
        epochs=1,
    ):
        """
        train llm model on dataset with llama-2 format
        """
        # load data
        train_datasets, test_datasets = self.load_data(
            extracted_label=extracted_label,
            normalize=normalize,
            downsampling_n_instances=downsampling_n_instances,
            downsampling_n_instances_train=downsampling_n_instances_train,
            downsampling_n_instances_test=downsampling_n_instances_test,
            name_feature=name_feature,
            save=save,
        )

        # load model, tokenizer and lora config
        model = self.load_model(quantized=quantized)
        tokenizer = self.load_tokenizer()
        peft_config = self.load_peft_config(
            lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
        )

        # optimizer
        if quantized:
            optim = "paged_adamw_32bit"
        else:
            optim = "adamw_torch"

        # training arguments
        logging_step = 1
        num_train_epichs = 1
        output_dir = f"./results_{self.name_data}"
        group_by_length = True

        fp16 = True
        bf16 = False
        lr_scheduler_type = "constant"

        training_arguments = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            logging_steps=logging_step,
            eval_steps=logging_step,
            learning_rate=learning_rate,
            group_by_length=group_by_length,
            lr_scheduler_type="constant",
            fp16=fp16,
            bf16=bf16,
            report_to="wandb",
            seed=self.seed,
            run_name=f"llama-lora-{self.name_data}",
        )

        self.training_loop(
            model=model,
            epochs=epochs,
            train_dataset=train_datasets,
            test_dataset=test_datasets,
            peft_config=peft_config,
            tokenizer=tokenizer,
            training_arguments=training_arguments,
        )

    def training_loop(
        self,
        model,
        epochs,
        train_dataset,
        test_dataset,
        peft_config,
        tokenizer,
        training_arguments: TrainingArguments,
    ):
        """
        training loop
        """
        # for ep in range(epochs):

        # print("epoch", ep)

        # train the model
        model.train()
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            tokenizer=tokenizer,
            args=training_arguments,
        )
        trainer.train()  # resume_from_checkpoint=True

        # evaluation
        model.eval()
        self.evaluation(model=model, datasets=train_dataset, tokenizer=tokenizer)
        self.evaluation(model=model, datasets=test_dataset, tokenizer=tokenizer)

    def evaluation(
        self,
        model,
        datasets,
        tokenizer,
    ):
        """
        evaluation mode, only for check accuracy
        """
        # total instance
        y_true_total = []
        y_pred_total = []

        # dataloader for faster training
        dataloader = DataLoader(dataset=datasets, batch_size=8, shuffle=False)

        # model in evaluation mode
        model.eval()

        start = default_timer()
        with torch.inference_mode():
            for idx, batch in tqdm(
                enumerate(dataloader), desc="Processing", unit="batch"
            ):
                question_prompts = batch["question"]  # Get batch of questions
                answers = batch["answer"]

                # Tokenize entire batch at once (instead of looping one-by-one)
                inputs = tokenizer(
                    question_prompts, return_tensors="pt", padding=True, truncation=True
                ).to(self.device)

                # Generate responses for entire batch at once
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    top_p=1.0,
                )

                # Decode responses
                responses = tokenizer.batch_decode(
                    output_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                print("responses:", responses)

                y_true = self.extract_label_from_response(list_strings=responses)
                y_true_total = y_true_total + y_true

                y_pred = self.extract_label_from_response(list_strings=answers)
                y_pred_total = y_pred_total + y_pred

                # Print results
                for idx, response in enumerate(responses):
                    print(f"Sample {idx}:")
                    print("Question:", question_prompts[idx])
                    print("Response:", response)
                    print("Answer:", batch["answer"][idx])

        print("y_true_total:", y_true_total)
        print("y_pred_total:", y_pred_total)
        accuracy = self.calculate_accuracy_y_text_list(
            y_true=y_true_total, y_pred=y_true_total
        )
        print("accuracy:", accuracy)
        end = default_timer()
        duration = end - start
        print("duration:", duration)

    def extract_label_from_response(self, list_strings):
        """
        use re to extract the label from the list of response or answer
        """
        # Regular expression to match "Y =" followed by any number (integer or float)
        list_match = [re.search(r"Y\s*=\s*(-?\d+(\.\d+)?)", i) for i in list_strings]
        list_y = [i.group() for i in list_match]
        return list_y

    def calculate_accuracy_y_text_list(self, y_true, y_pred):
        """
        calculate accuracy
        """
        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        accuracy = (correct / len(y_true)) * 100
        return accuracy


# run this script
if __name__ == "__main__":

    name_data = "HST"
    seed = 1998

    def set_random_seed(seed=42):
        """
        Sets the random seed for reproducibility in PyTorch, NumPy, and Python's random module.
        Also ensures CUDA determinism if GPU is available.
        """
        random.seed(seed)  # Python random seed
        np.random.seed(seed)  # NumPy random seed
        torch.manual_seed(seed)  # PyTorch random seed
        transformers.set_seed(seed)

        # Check if CUDA is available
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)  # Set CUDA seed
            torch.cuda.manual_seed_all(seed)  # If multi-GPU, set for all GPUs
            torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
            torch.backends.cudnn.benchmark = (
                False  # Disable CUDNN benchmarking for reproducibility
            )

    set_random_seed(seed=seed)

    llm_hst = LLM(name_data=name_data, seed=seed)

    llm_hst.classification(quantized=False)
