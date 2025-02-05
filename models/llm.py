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
from peft import LoraConfig
from trl import SFTTrainer
import sys
import os
import torch
from sklearn.metrics import accuracy_score

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
        lora_r,
        lora_alpha,
        lora_dropout,
        extracted_label=None,
        normalize=False,
        downsampling_n_instances=None,
        downsampling_n_instances_train=None,
        downsampling_n_instances_test=None,
        name_feature=False,
        save=False,
        quantized=False,
        batch_size=32,
        gradient_accumulation_steps=1,
        learning_rate=0.0001,
        weight_decay=0.001,
        epochs=10,
    ):
        """
        train llm model on dataset with llama-2 format
        """
        # load data
        train_datasets, test_datasets = self.load_data(
            lora_r,
            lora_alpha,
            lora_dropout,
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
        output_dir = "./results"
        group_by_length = True

        training_arguments = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epichs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            logging_steps=logging_step,
            learning_rate=learning_rate,
            group_by_length=group_by_length,
            report_to="wandb",
            run_name="llama-lora-experiment",
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
        for ep in epochs:

            # train the model
            model.train()
            trainer = SFTTrainer(
                model=model,
                train_dataset=train_dataset,
                peft_config=peft_config,
                dataset_text_field="text",
                tokenizer=tokenizer,
                args=training_arguments,
            )

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
        model.eval()
        with torch.no_grad():
            for i in datasets:
                question_prompt = i["question"]
                print("question_prompt:", question_prompt)
                inputs = tokenizer(question_prompt, return_tensors="pt").to(self.device)
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=500,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0
                )
                response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                print("response:", response)
                answer = i["answer"]
                print("answer:", answer)


# run this script
if __name__ == "__main__":

    name_data = "HST"
    seed = 1998
    llm_hst = LLM(name_data=name_data, seed=seed)
    path_models_directory = llm_hst.path_models_directory
    print("path_models_directory:", path_models_directory)
