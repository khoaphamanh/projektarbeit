import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
import re
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# # load dataset
# name_dataset = "timdettmers/openassistant-guanaco"
# dataset = load_dataset(name_dataset)
# # print("dataset:", dataset)
# train_data = dataset["train"]

# # print("train_data len:", len(train_data))
# # print("train_data shape:", train_data.shape)
# print("train_data [0]:", train_data[0])

# train_data[0]["text"] = "hello"

# print("train_data:", train_data[0])


# Load the dataset
name_dataset = "timdettmers/openassistant-guanaco"
dataset = load_dataset(name_dataset)
train_data = dataset["train"]
