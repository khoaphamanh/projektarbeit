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
            compute_dtype = torch.float16()
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

    def train(self):
        """
        train llm model on dataset with llama-2 format
        """


# run this script
if __name__ == "__main__":

    name_data = "HST"
    seed = 1998
    llm = LLM(name_data=name_data, seed=seed)
    path_models_directory = llm.path_models_directory
    print("path_models_directory:", path_models_directory)
