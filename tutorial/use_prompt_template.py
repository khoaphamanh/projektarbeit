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

# prompt template
system_behavior = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
start_sentence_token = "<s>"
end_sentence_token = "</s>"
start_user_prompt = "[INST]"
end_user_prompt = "[/INST]"


def prompt_template(example):
    # each text ist value of the key text in each dict
    text = example["text"]

    # split each text with human ask and assistant answer
    split_text = re.split(r"(### Human:|### Assistant:)", text)
    human_text = [
        split_text[i + 1].strip()
        for i in range(1, len(split_text), 2)
        if "Human" in split_text[i]
    ]
    assistant_text = [
        split_text[i + 1].strip()
        for i in range(1, len(split_text), 2)
        if "Assistant" in split_text[i]
    ]

    result = []
    for i, (h, a) in enumerate(zip(human_text, assistant_text)):

        result.append(start_sentence_token)
        result.append(start_user_prompt)
        if i == 0:
            result.append("<<SYS>> {} <</SYS>>".format(system_behavior))
        result.append(h)
        result.append(end_user_prompt)
        result.append(a)
        result.append(end_sentence_token)

    result = " ".join(result)

    # save the result to key "text"
    example["text"] = result

    return example


# Print the first instance before modification
print("Before modification:", train_data[0])
print("Before modification:", train_data[1])

# apply the prompt template to train_data
train_data = train_data.map(prompt_template)

# Print the modified instances
print("After modification:", train_data[0])
print("After modification:", train_data[1])
