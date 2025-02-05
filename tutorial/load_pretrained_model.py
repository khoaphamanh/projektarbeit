"""
eval model from newmodel directory
"""

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
from train_llm import model_name, dataset_name, new_model, device_map

# load the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
"""
AutoTokenizer là class chung để sử dụng tokenizer nếu như ng dung k biết nên sử dụng class tokenizer nào cho model llm của mình. Tokenizer không được tính là model weights, chúng đc predefine và được tính là
"""
# Set pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Set padding direction
tokenizer.padding_side = "right"

# load pretrained model
model = AutoModelForCausalLM.from_pretrained(new_model, device_map=device_map)

# Run text generation pipeline with our next model
# pipeline bao gồm high-level function làm tắt toàn bộ các bước:
# từ text tới đưa về syntax của llama
# tokenizer text
# forward pass in evaluation mode in model
# get the result

prompt = "What is a large language model?"
pipe = pipeline(
    task="text-generation", model=model, tokenizer=tokenizer, max_length=200
)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]["generated_text"])


# do it manually
model.eval()
formatted_input = f"<s>[INST] {prompt} [/INST]"
print("formatted_input:", formatted_input)

# Tokenize the input text
inputs = tokenizer(formatted_input, return_tensors="pt").to(device)

# Generate the model's response
with torch.no_grad():
    output_ids = model.generate(
        **inputs, max_new_tokens=200, temperature=0.7, top_k=50, top_p=0.9
    )

# Decode the generated output into text
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Model Response:", response)

"""
eval model from result directory and let model always give the same answer each time
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Define the paths
base_model_name = "meta-llama/Llama-2-7b-chat-hf"  # Base model name
checkpoint_dir = "./results/checkpoint-250"  # Directory with the specific checkpoint

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load the base model
device = "cuda" if torch.cuda.is_available() else "cpu"
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto")

# Load the LoRA adapter from the checkpoint
model = PeftModel.from_pretrained(
    base_model,
    checkpoint_dir,
    torch_dtype=torch.float16,  # Match the precision of the base model.
    device_map="auto",  # Ensure consistent device placement.
)

# Move the model to evaluation mode
model.eval()

# Example: Generate text
prompt = "Kể cho tôi lịch sử của thành phố Đắk Lắk"
formatted_input = f"<s>[INST] {prompt} [/INST]"
inputs = tokenizer(formatted_input, return_tensors="pt").to(device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=False,  # No randomness
        temperature=0.0,  # Force deterministic output
        top_p=1.0,
    )

# Decode the generated text
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Model Response:", response)
