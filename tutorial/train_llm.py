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

# the model used in this tutorial
model_name = "meta-llama/Llama-2-7b-chat-hf"

# dataset with 1k samples
dataset_name = "mlabonne/guanaco-llama2-1k"

# our new pretrained fine tunned model using LoRA
new_model = "Llama-2-7b-chat-finetune"

# hyperparameters for LoRA
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1

# bit and bytes parameters
"""
4 bit precision aps dụng cho mỗi component trong llm, chọn max và min value cho mỗi weights và sau đó
chia cái range (min, max) thành 2^4 = 16 giá trị riêng biệt và đối với mỗi giá trị trong weights, chúng 
sẽ được rounded cho giá trị gần nhất. W_quntized là bao gồm các index của giá trị trong range gần nhất vs giá trị gốc của matrix.
"""
use_4bit = True

# Compute dtype for 4-bit base models
"""
kiểu dữ liệu dtype của weights (sau khi đã được ứng dung 4 bit precision), tại vì trong quá trình tính toán
các ma trận weights nhân với nhau sẽ được lưu trữ kết quả ở float16
"""
compute_dtype = "float16"

# Qunatization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"
"""
cách để split các range trong weights khi dùng 4 bit precision thành 16 unique values 
fb4 (floating point 4-bit): chia đều trong khoảng min max value
nf4 (normalized floating point 4-bit): normalized wieghts (giá trị được chia) = weight / sqrt(variance of weights)
"""

# output directory, đây là nơi lưu lại metrcis, những thứ đã được train như optimizer, giúp cho việc resume training dễ dàng hơn.
output_dir = "./results"

# Enable fp16/bf16 training (set bf16 to True with an A100)
"""
This line configures whether the model should use FP16 (16-bit floating-point) or BF16 (16-bit brain floating-point) precision for training.
"""
fp16 = False
bf16 = True if torch.cuda.is_bf16_supported() else False

# training process hyperparameters
epochs = 1
batch_size = 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
"""
gradient checkpoint là không lưu trữ giá trị zwischen result trong quá trình forward để đỡ tốn dung lượng
các giá trị zwischen result này sẽ được tính lại trong quá trình backward
vd: a1 = x*w +b
a2 = a1*w +b
y =  a2*w +b
nếu chỉ có giá trị a1 dcd lưu thì trong quá trình backward từ y tới a2, a2 sẽ được tính lại từ a1 (giá trị đã đc lưu)
"""
max_gradient_norm = 0.3
"""
giá trị làm thay đổi gradien dL
dL = dL * max_gradient_norm / gradient_norm
gradient_norm = sqrt(graident^2)
"""
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
"""
một biến thể của adamw với việc dùng 32 bit 
"""
warmup_ratio = 0.03
"""
warmup_ratio ám chỉ trong khoảng từ step 0 tới 3% của tổng step, lr sẽ đi từ 0 tới cực đại theo giá trị linear
sau đó lr sẽ có hình dạng cosine the công thức lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * t / T))
"""
lr_scheduler_type = "cosine"
group_by_length = True
"""
trong 1 batch sẽ có những sequences có cùng lenght để đỡ tốn memory cho padding
"""
save_steps = 0
"""
cứ bn step thì lưu lại weights, bias hay các giá trị có trong optimizer nhằm tiếp tục train, resume training. ( trường hợp này là 0)
"""
logging_steps = 25
"""
sau bn step thì lưu current loss, learning rate và training step
"""
max_seq_length = None
packing = False
"""
packing ám chỉ ghép 2 câu ngắn vào 1 câu, giúp đỡ tốn tài nguyên. vd:
Sequence 1: [This, is, short]
Sequence 2: [Another, one]
Packed input (if packing=True): [This, is, short, [SEP], Another, one]
"""
device_map = "auto"

# load dataset
dataset = load_dataset(dataset_name, split="train")

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, compute_dtype)  # torch.float16

if __name__ == "__main__":

    """
    trong bài này sẽ sử dụng QLoRA về bản chất là kết hợp của
    - Quantizer: các weights sẽ được quantizer hay đưa về dạng 4 bit precision
    - Ứng dung LoRA vào quá trình huấn luyện
    """

    # QLoRA configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map=device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    """
    AutoTokenizer là class chung để sử dụng tokenizer nếu như ng dung k biết nên sử dụng class tokenizer nào cho model llm của mình. Tokenizer không được tính là model weights, chúng đc predefine và được tính là 
    """
    # Set pad_token to eos_token
    tokenizer.pad_token = tokenizer.eos_token

    # Set padding direction
    tokenizer.padding_side = "right"

    # load LoRA config
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",  # chọn bias bằng none thì bias sẽ k xuất hiện trong A và B, bias gốc trong model cũng sẽ không đc update, tương tự như weights.
        task_type="CAUSAL_LM",  # use case: Text generation (e.g., writing stories, generating code).
    )

    # Set training parameters, TrainingArguments define the training configuraion for training process
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_gradient_norm,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        # report_to="tensorboard",
    )

    # Check column names
    print(dataset.column_names)

    # Alternatively, you can also print the features:
    print(dataset.features)

    # train the model using SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",  # dataset có dict gồm key là "text"
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )

    trainer.train()
    trainer.model.save_pretrained(new_model)
