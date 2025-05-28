import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import argparse
from llm import LLM

# argument parser
parser = argparse.ArgumentParser(description="Load and run a pre-trained model.")
parser.add_argument(
    "-d",
    "--data_name",
    type=str,
    help="choose data name between 'HST' or 'TEP', default is 'HST'",
    choices=["HST", "TEP"],
    default="HST",
)

args = parser.parse_args()

# set data name
if args.data_name == "HST":
    name_data = "HST"
elif args.data_name == "TEP":
    name_data = "TEP"
print("name_data:", name_data)

# init llm
seed = 1998
llm_run = LLM(name_data=name_data, seed=seed)

# path to pretrained model
directory_pretrained_model = f"results_{name_data}"
path_models_directory = llm_run.path_models_directory
path_pretrained_model = os.path.join(path_models_directory, directory_pretrained_model)
checkpoint_files = [i for i in os.listdir(path_pretrained_model) if "checkpoint" in i][
    0
]
path_checkpoint = os.path.join(path_pretrained_model, checkpoint_files)

# tokenizer
tokenizer = llm_run.load_tokenizer()

# load model
base_model = llm_run.load_model(quantized=True)
# model = AutoModelForCausalLM.from_pretrained(path_checkpoint, device_map="auto")
# Load base LLaMA-2 model
# base_model = AutoModelForCausalLM.from_pretrained(
#     "meta-llama/Llama-2-7b-chat-hf",  # or your base model path
#     device_map="auto",
# )

# Load LoRA adapter (your checkpoint)
model = PeftModel.from_pretrained(base_model, path_checkpoint)
# # apply LoRA to model
# model = PeftModel.from_pretrained(model, path_checkpoint)
model.eval()

# prompt
if name_data == "HST":
    prompt = "Tell me if the value of Y is 0 or 1. If the features are: 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5000, 0.5254, 0.5163, 0.5137, 0.9719, 0.9886, 0.9111, 0.4220, 0.6667, 0.5455, 0.0000, 0.0000, 0.0000, 1.0000, 0.9661, 0.9661, 1.0000, 0.8538, 0.0000, 0.0000, 1.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0625, 0.0000, 0.0000, 0.0000, 1.0000. What should be Y?"
    # Y = 1
elif name_data == "TEP":
    prompt = "Tell me if the value of Y is 0 or 1 or 4 or 5. If the features are: 1.0000, 83.0000, 0.9257, 3666.8000, 4356.9000, 8.2511, 26.9010, 41.9320, 2682.4000, 76.7330, 120.3800, 0.3046, 79.9720, 51.5360, 2607.7000, 24.1570, 49.1490, 3091.1000, 23.7010, 70.9100, 363.3400, 326.2500, 95.0840, 76.3400, 30.3080, 9.0524, 26.8640, 6.8916, 20.0120, 1.6396, 30.2820, 14.0840, 24.5430, 1.4694, 20.0290, 2.2003, 4.7404, 2.1987, 0.0303, 0.9939, 0.1107, 54.2140, 43.2380, 61.6550, 52.3680, 91.0190, 55.0690, 18.8310, 37.4680, 42.6210, 44.5650, 89.1600, 39.0930, 17.4510. What should be Y?"
    # Y = 1

prompt = (
    prompt
    + " Can you also give me the feature importance? If so, please provide the feature importance in a list format, and tell me which feature is the most important"
)


question = f"<s>[INST] {prompt} [/INST]"

# create inputs for the model
inputs = tokenizer(question, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs, max_new_tokens=2000, do_sample=False, temperature=0.7, top_p=1.0
    )
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("response:", response)
