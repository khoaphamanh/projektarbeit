from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    Trainer,
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
import wandb
from datetime import datetime
from sklearn.metrics import accuracy_score
import re

# Set environment variable before torch is imported
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
        self.gpus = []
        self.total_vram = 0
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            for i in range(n_gpus):
                props = torch.cuda.get_device_properties(i)
                gpu_info = {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "vram_gb": round(props.total_memory / 1024 / 1024 / 1024, 2),
                }
                self.gpus.append(gpu_info)
                self.total_vram = self.total_vram + round(
                    props.total_memory / 1024 / 1024 / 1024, 2
                )
        else:
            self.gpus = None

        self.n_gpus = len(gpu_info) if self.gpus is not None else 0

        print("self.gpus:", self.gpus)

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

    def load_data_llm(
        self,
        extracted_label=None,
        normalize=False,
        downsampling_n_instances=None,
        downsampling_n_instances_train=None,
        downsampling_n_instances_test=None,
        name_feature=False,
        save_data=False,
    ):
        """
        load data in llama-2 format
        """
        train_datasets, test_datasets = self.load_data(
            extracted_label=extracted_label,
            normalize=normalize,
            downsampling_n_instances=downsampling_n_instances,
            downsampling_n_instances_train=downsampling_n_instances_train,
            downsampling_n_instances_test=downsampling_n_instances_test,
            name_feature=name_feature,
            convert_to_text=True,
            save_data=save_data,
        )

        return train_datasets, test_datasets

    def create_name_run(self, hpo=False, index_trial=None, index_split=None):
        """
        create name based on datetime and name data
        """
        # name of the run for not hpo case
        if not hpo:
            name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            name = f"{self.name_data}_{name}"

        # name of the run for hpo case
        else:
            name = f"{self.name_data}_trial_{index_trial}_split_{index_split}"

        return name

    def hyperparameters_configurations_dict(self, print_out=False, **kwargs):
        """
        Return a hyperparameter dictionary and optionally print it with sorted keys.
        """
        if print_out:
            for k in sorted(kwargs.keys()):
                print(f"{k}: {kwargs[k]}")
        return kwargs

    def preparation_training_process(
        self,
        train_datasets,
        test_datasets,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        extracted_label=None,
        normalize=False,
        downsampling_n_instances=None,
        downsampling_n_instances_train=None,
        downsampling_n_instances_test=None,
        name_feature=False,
        save_data=False,
        quantized=False,
        batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=0.0001,
        max_new_tokens=5,
        save_model=False,
        epochs=10,
        n_splits=None,
        index_split=None,
        hpo=False,
    ):
        """
        preparation all things after load data and before the training process
        """

        # load model, tokenizer and lora config
        model = self.load_model(quantized=quantized)
        for name, param in model.named_parameters():
            print(f"Parameter: {name}, Device: {param.device}")

        # optimizer
        if quantized:
            optim = "paged_adamw_32bit"
        else:
            optim = "adamw_torch"

        # print the params
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # some default hyperparameters
        name_run = self.create_name_run()
        project = (
            "projektarbeit_khoa_quy"
            if not hpo
            else f"projektarbeit_cv_{self.name_data}"
        )

        n_instances_train = len(train_datasets)
        n_instances_test = len(test_datasets)

        logging_step = np.ceil(
            len(train_datasets) / (batch_size * gradient_accumulation_steps)
        )
        save_strategy = "no" if not save_model else "epoch"
        output_dir = f"./results_{self.name_data}"
        group_by_length = True
        fp16 = True
        bf16 = False
        logging_strategy = "epoch"

        dict_hyperparameters = self.hyperparameters_configurations_dict(
            # hyperparameters model
            learning_rate=learning_rate,
            quantized=quantized,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            epochs=epochs,
            name_pretrained_model=self.name_model_llama_2,
            total_params=total_params,
            trainable_params=trainable_params,
            n_splits=n_splits,
            index_split=index_split,
            # hyperparameter data
            normalize=normalize,
            name_data=self.name_data,
            extracted_label=extracted_label,
            downsampling_n_instances=downsampling_n_instances,
            downsampling_n_instances_train=downsampling_n_instances_train,
            downsampling_n_instances_test=downsampling_n_instances_test,
            name_feature=name_feature,
            save_data=save_data,
            n_instances_train=n_instances_train,
            n_instances_test=n_instances_test,
            n_batch_train=np.ceil(n_instances_train / batch_size),
            n_batch_test=np.ceil(n_instances_test / batch_size),
            # configurations
            n_gpus=self.n_gpus,
            seed=seed,
            name_run=name_run,
            logging_step=logging_step,
            project=project,
            save_strategy=save_strategy,
            output_dir=output_dir,
            group_by_length=group_by_length,
            fp16=fp16,
            bf16=bf16,
            logging_strategy=logging_strategy,
            max_new_tokens=max_new_tokens,
            # print out
            print_out=True,
        )
        gpus_id = {f"gpu{i+1}": gpu for i, gpu in enumerate(self.gpus)}
        dict_hyperparameters = {**dict_hyperparameters, **gpus_id}

        return model, dict_hyperparameters

    def classification(
        self,
        project="projektarbeit_khoa_quy",
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        extracted_label=None,
        normalize=False,
        downsampling_n_instances=None,
        downsampling_n_instances_train=None,
        downsampling_n_instances_test=None,
        name_feature=False,
        save_data=False,
        quantized=False,
        batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=0.0001,
        max_new_tokens=5,
        save_model=False,
        epochs=10,
        trial=None,
        n_splits=None,
        index_split=None,
        hpo=False,
    ):
        """
        train llm model on dataset with llama-2 format
        """
        # control batchsize for suitable vram
        if self.name_data == "TEP":
            if self.total_vram < 11:
                batch_size = 1
            elif self.total_vram < 24:
                batch_size = 3
            else:
                batch_size = 6

        elif self.name_data == "HST":
            if self.total_vram < 11:
                batch_size = 1
            elif self.total_vram < 24:
                batch_size = 3
            else:
                batch_size = 8

        print("batch_size:", batch_size)

        # load data
        train_datasets, test_datasets = self.load_data_llm(
            extracted_label=extracted_label,
            normalize=normalize,
            downsampling_n_instances=downsampling_n_instances,
            downsampling_n_instances_train=downsampling_n_instances_train,
            downsampling_n_instances_test=downsampling_n_instances_test,
            name_feature=name_feature,
            save_data=save_data,
        )

        # load model, tokenizer and lora config
        model, dict_hyperparameters = self.preparation_training_process(
            train_datasets=train_datasets,
            test_datasets=test_datasets,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            extracted_label=extracted_label,
            normalize=normalize,
            downsampling_n_instances=downsampling_n_instances,
            downsampling_n_instances_train=downsampling_n_instances_train,
            downsampling_n_instances_test=downsampling_n_instances_test,
            name_feature=name_feature,
            save_data=save_data,
            quantized=quantized,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            max_new_tokens=max_new_tokens,
            save_model=save_model,
            epochs=epochs,
            n_splits=n_splits,
            index_split=index_split,
            hpo=hpo,
        )

        # load tokenizer and peft config
        tokenizer = self.load_tokenizer()
        peft_config = self.load_peft_config(
            lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
        )
        for name, param in model.named_parameters():
            print(f"Parameter: {name}, Device: {param.device}")

        # optimizer
        if quantized:
            optim = "paged_adamw_32bit"
        else:
            optim = "adamw_torch"

        # print the params
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # init wandb
        name = self.create_name_run()
        hyperparameters_model = self.hyperparameters_configurations_dict(
            learning_rate=learning_rate,
            quantized=quantized,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            epochs=epochs,
            print_out=True,
        )

        configurations = {f"gpu{i+1}": gpu for i, gpu in enumerate(self.gpus)}
        configurations["n_gpus"] = self.n_gpus
        configurations["n_params"] = total_params
        configurations["n_params_trainable"] = trainable_params
        configurations["seed"] = self.seed
        configurations["name_model"] = self.name_model_llama_2
        configurations = self.hyperparameters_configurations_dict(
            **configurations, print_out=True
        )

        n_instances_train = len(train_datasets)
        n_instances_test = len(test_datasets)
        hyperparameters_data = self.hyperparameters_configurations_dict(
            name_data=self.name_data,
            extracted_label=extracted_label,
            normalize=normalize,
            downsampling_n_instances=downsampling_n_instances,
            downsampling_n_instances_train=downsampling_n_instances_train,
            downsampling_n_instances_test=downsampling_n_instances_test,
            name_feature=name_feature,
            save=save_data,
            n_instances_train=n_instances_train,
            n_instances_test=n_instances_test,
            n_batch_train=np.ceil(n_instances_train / batch_size),
            n_batch_test=np.ceil(n_instances_test / batch_size),
            print_out=True,
        )

        wandb.init(
            project=project,
            name=name,
            settings=wandb.Settings(
                code_dir=os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..")
                )  # self.project_root_dir  # os.path.dirname(self.path_models_directory)
            ),
            config={
                "aa_hyperparameters_model": hyperparameters_model,
                "aa_configurations": configurations,
                "aa_hyperparameters_data": hyperparameters_data,
            },
            save_code=True,
        )

        # training arguments
        logging_step = 1
        logging_step = np.ceil(
            len(train_datasets) / (batch_size * gradient_accumulation_steps)
        )

        save_strategy = "no" if not save_model else "epoch"
        output_dir = f"./results_{self.name_data}"
        group_by_length = True

        fp16 = True
        bf16 = False
        lr_scheduler_type = "constant"  # "cosine_with_restarts" # "cosine"
        logging_strategy = "epoch"  #  "steps"  # "epoch"

        training_arguments = TrainingArguments(
            # configuration parameters
            output_dir=output_dir,
            report_to="wandb",
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            learning_rate=learning_rate,
            group_by_length=group_by_length,
            lr_scheduler_type=lr_scheduler_type,
            fp16=fp16,
            bf16=bf16,
            seed=self.seed,
            logging_strategy=logging_strategy,
            save_strategy=save_strategy,
            # train parameters
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            logging_steps=logging_step,
            # val parameters
            eval_steps=logging_step,
            eval_strategy=logging_strategy,
            eval_accumulation_steps=gradient_accumulation_steps,
        )

        self.training_loop(
            model=model,
            epochs=epochs,
            train_dataset=train_datasets,
            test_dataset=test_datasets,
            peft_config=peft_config,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
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
        max_new_tokens,
        training_arguments: TrainingArguments,
    ):
        """
        training loop
        """

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
            compute_metrics=None,  # self.custom_accuracy,
            callbacks=[
                AccuracyCallback(self, max_new_tokens, train_dataset, mode="train"),
                AccuracyCallback(self, max_new_tokens, test_dataset, mode="test"),
            ],
        )
        trainer.train()  # resume_from_checkpoint=True

        self.evaluation(
            model=model,
            datasets=train_dataset,
            tokenizer=tokenizer,
            batch_size=training_arguments.per_device_train_batch_size * 6,
            max_new_token=max_new_tokens,
        )
        self.evaluation(
            model=model,
            datasets=test_dataset,
            tokenizer=tokenizer,
            max_new_token=max_new_tokens,
            batch_size=training_arguments.per_device_train_batch_size * 6,
        )

        trainer.evaluate()

    def evaluation(self, model, datasets, tokenizer, batch_size, max_new_token):
        """
        evaluation mode, only for check accuracy
        """
        # total instance
        y_true_total = []
        y_pred_total = []

        # dataloader for faster training
        dataloader = DataLoader(dataset=datasets, batch_size=batch_size, shuffle=False)

        # model in evaluation mode
        model.eval()

        start = default_timer()
        with torch.inference_mode():
            for idx, batch in tqdm(
                enumerate(dataloader), desc="Processing", unit="batch"
            ):
                # get question and answer
                question_prompts = batch["question"]
                answers = batch["answer"]

                # Tokenize entire batch at once (instead of looping one-by-one)
                inputs = tokenizer(
                    question_prompts, return_tensors="pt", padding=True, truncation=True
                ).to(self.device)

                # Generate responses for entire batch at once
                output_ids = model.generate(
                    **inputs, max_new_tokens=max_new_token, do_sample=False, top_p=1.0
                )  # remove top_p = 1.0 because already use do_sample

                # Decode responses
                responses = tokenizer.batch_decode(
                    output_ids,
                    skip_special_tokens=True,
                )
                print("responses:", responses)

                y_true = self.extract_label_from_response(answers)
                y_true_total = y_true_total + y_true

                y_pred = self.extract_label_from_response(responses)
                y_pred_total = y_pred_total + y_pred

                # Print results
                for idx, response in enumerate(responses):
                    print(f"Sample {idx}:")
                    print("Question:", question_prompts[idx])
                    print("Response:", response)
                    print("Answer:", batch["answer"][idx])

        print("y_true_total:", y_true_total)
        print("y_pred_total:", y_pred_total)

        # calculate accuracy score
        accuracy = accuracy_score(y_pred=y_pred_total, y_true=y_true_total)
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
        list_y = [i.group() if i else 99 for i in list_match]

        return list_y

    def class_wise_accuracy(self, y_true, y_pred):
        """
        calculate accuracy each class
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        unique_labels = np.unique(y_true)
        class_accuracy = {}

        for label in unique_labels:
            # Mask for samples belonging to this class
            mask = y_true == label
            correct = np.sum(y_pred[mask] == y_true[mask])
            total = np.sum(mask)
            accuracy = correct / total if total > 0 else 0.0
            class_accuracy[label] = accuracy

        return class_accuracy


class AccuracyCallback(TrainerCallback):
    def __init__(self, llm_instance, max_new_tokens, dataset, mode="train"):
        self.llm = llm_instance
        self.max_new_tokens = max_new_tokens
        self.dataset = dataset
        self.mode = mode
        self.last_accuracy = None
        self.last_loss = None

    def on_epoch_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        print(
            f"\n[TrainAccuracyCallback] Epoch {state.epoch} ended. Computing training accuracy and loss..."
        )

        tokenizer = self.llm.load_tokenizer()
        model = kwargs["model"]
        device = self.llm.device
        batch_size = args.per_device_train_batch_size * 6

        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)

        y_true_total = []
        y_pred_total = []
        losses = []

        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                # --- LOSS computation using full text (prompt + answer) ---
                if "text" not in batch:
                    raise ValueError(
                        "Batch must contain 'text' field for loss computation."
                    )
                full_texts = batch["text"]
                encoded = tokenizer(
                    full_texts, return_tensors="pt", padding=True, truncation=True
                ).to(device)
                labels = encoded.input_ids.clone()
                loss_output = model(**encoded, labels=labels)
                losses.append(loss_output.loss.item())

                # --- Accuracy computation using question only ---
                questions = batch["question"]
                answers = batch["answer"]

                inputs = tokenizer(
                    questions, return_tensors="pt", padding=True, truncation=True
                ).to(device)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    top_p=1.0,
                )

                responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                y_pred = self.llm.extract_label_from_response(responses)
                y_true = self.llm.extract_label_from_response(answers)

                y_true_total += y_true
                y_pred_total += y_pred

                for idx, response in enumerate(responses):
                    print(f"Sample {idx}:")
                    print("Question:", questions[idx])
                    print("Response:", response)
                    print("Answer:", answers[idx])

        # Final metrics
        self.last_loss = np.mean(losses)
        self.last_accuracy = accuracy_score(y_pred=y_pred_total, y_true=y_true_total)

        print(
            f"[AccuracyCallback] Epoch {state.epoch}: {self.mode} loss: {self.last_loss:.4f}"
        )
        print(
            f"[AccuracyCallback] Epoch {state.epoch}: {self.mode} accuracy: {self.last_accuracy:.4f}"
        )

        if wandb.run:
            wandb.log(
                {
                    f"{self.mode}_custom_accuracy": self.last_accuracy,
                    f"{self.mode}_custom_loss": self.last_loss,
                    "epoch": state.epoch,
                },
                step=state.global_step,
            )

        acc_each_class = self.llm.class_wise_accuracy(y_true_total, y_pred_total)
        for label, acc_class in acc_each_class.items():
            print(f"Accuracy for class {label}: {acc_class:.4f}")
            if wandb.run:
                wandb.log(
                    {f"class_{label}_custom_accuracy": acc_class, "epoch": state.epoch},
                    step=state.global_step,
                )


# run this script
if __name__ == "__main__":

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

    # # init llm hst
    # name_data = "TEP"  # "HST"
    # print("name_data:", name_data)
    # llm_run = LLM(name_data=name_data, seed=seed)

    # # hyperparameters
    # project = "projektarbeit_khoa_quy"
    # lora_r = 8
    # lora_alpha = 32
    # lora_dropout = 0.1
    # extracted_label = [0, 1, 4, 5]
    # normalize = True
    # downsampling_n_instances = None
    # downsampling_n_instances_train = 400
    # downsampling_n_instances_test = 160
    # name_feature = False
    # save = False
    # quantized = True
    # batch_size = 1
    # gradient_accumulation_steps = 1
    # learning_rate = 0.001
    # max_new_tokens = 5
    # save_model = False
    # epochs = 150

    # init llm hst
    name_data = "HST"
    print("name_data:", name_data)
    llm_run = LLM(name_data=name_data, seed=seed)

    # hyperparameters
    lora_r = 8
    lora_alpha = 32
    lora_dropout = 0.1
    extracted_label = None
    normalize = True
    downsampling_n_instances = 50
    downsampling_n_instances_train = None
    downsampling_n_instances_test = None
    name_feature = False
    save = False
    quantized = True
    batch_size = 1
    gradient_accumulation_steps = 1
    learning_rate = 0.001
    max_new_tokens = 5
    save_model = False
    epochs = 15

    llm_run.classification(
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        extracted_label=extracted_label,
        normalize=normalize,
        downsampling_n_instances=downsampling_n_instances,
        downsampling_n_instances_train=downsampling_n_instances_train,
        downsampling_n_instances_test=downsampling_n_instances_test,
        name_feature=name_feature,
        save_data=save,
        quantized=quantized,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_new_tokens=max_new_tokens,
        save_model=save_model,
        epochs=epochs,
    )
