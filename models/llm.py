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
import optuna
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
        lr_scheduler_type="constant",
        max_new_tokens=5,
        save_model=False,
        epochs=10,
        n_splits=None,
        index_split=None,
        index_trial=None,
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
        name_run = self.create_name_run(
            hpo=hpo, index_split=index_split, index_trial=index_trial
        )
        project = (
            "projektarbeit_khoa_quy"
            if not hpo
            else f"projektarbeit_cv_{self.name_data}"
        )

        n_instances_train = len(train_datasets)
        n_instances_test = len(test_datasets)

        logging_step_train = np.ceil(
            len(train_datasets) / (batch_size * gradient_accumulation_steps)
        )
        logging_step_eval = np.ceil(
            len(test_datasets) / (batch_size * gradient_accumulation_steps)
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
            lr_scheduler_type=lr_scheduler_type,
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
            seed=self.seed,
            name_run=name_run,
            logging_step_train=logging_step_train,
            logging_step_eval=logging_step_eval,
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
        index_trial=None,
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
            index_trial=index_trial,
            hpo=hpo,
        )

        # load tokenizer and peft config
        tokenizer = self.load_tokenizer()
        peft_config = self.load_peft_config(
            lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
        )

        # optimizer
        optim = dict_hyperparameters["optim"]

        # init wandb
        project = dict_hyperparameters["project"]
        name_run = dict_hyperparameters["name_run"]
        wandb.init(
            project=project,
            name=name_run,
            settings=wandb.Settings(
                code_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            ),
            config={
                "aa_dict_hyperparameters": dict_hyperparameters,
            },
            save_code=True,
        )

        # training arguments
        logging_step_train = dict_hyperparameters["logging_step_train"]
        logging_step_eval = dict_hyperparameters["logging_step_eval"]

        save_strategy = dict_hyperparameters["save_strategy"]
        output_dir = dict_hyperparameters["output_dir"]
        group_by_length = dict_hyperparameters["logging_step_eval"]

        fp16 = dict_hyperparameters["fp16"]
        bf16 = dict_hyperparameters["bf16"]

        lr_scheduler_type = dict_hyperparameters[
            "lr_scheduler_type"  # "cosine_with_restarts" # "cosine"
        ]
        logging_strategy = dict_hyperparameters["logging_strategy"]

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
            logging_steps=logging_step_train,
            # val parameters
            eval_steps=logging_step_eval,
            eval_strategy=logging_strategy,
            eval_accumulation_steps=gradient_accumulation_steps,
            per_device_eval_batch_size=batch_size,
        )

        self.training_loop(
            model=model,
            train_dataset=train_datasets,
            test_dataset=test_datasets,
            peft_config=peft_config,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            training_arguments=training_arguments,
            trial=trial,
        )

    def training_loop(
        self,
        model,
        train_dataset,
        test_dataset,
        peft_config,
        tokenizer,
        max_new_tokens,
        training_arguments: TrainingArguments,
        trial=None,
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
                # MetricsCallback for train data
                MetricsCallback(
                    self, max_new_tokens, train_dataset, mode="train", trial=None
                ),
                # MetricsCallback for test data
                MetricsCallback(
                    self, max_new_tokens, test_dataset, mode="test", trial=trial
                ),
            ],
        )
        trainer.train()

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

    def compute_metrics_on_dataset(
        self, model, dataset, tokenizer, max_new_tokens, batch_size
    ):
        """
        function to calculate metrics like loss and accuracy after each epoch
        """
        # set model in eval mode
        model.eval()

        # convert data to dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # lists
        y_true_total = []
        y_pred_total = []
        losses = []

        with torch.inference_mode():
            for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):

                # input as text
                full_texts = batch["text"]

                # calculate the loss
                encoded = tokenizer(
                    full_texts, return_tensors="pt", padding=True, truncation=True
                ).to(self.device)
                labels = encoded.input_ids.clone()
                output = model(**encoded, labels=labels)
                losses.append(output.loss.item())

                # get the prediction as tensor
                inputs = tokenizer(
                    batch["question"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.device)
                outputs = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False, top_p=1.0
                )

                # prediction as text
                responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                # ground true
                answer = batch["answer"]
                y_pred = self.extract_label_from_response(responses)
                y_true = self.extract_label_from_response(answer)

                # add to list
                y_pred_total += y_pred
                y_true_total += y_true

        # print out the y list
        print("y_pred_total:", y_pred_total)
        print("y_true_total:", y_true_total)

        # return the metrics
        metrics = {
            "accuracy": accuracy_score(y_true_total, y_pred_total),
            "loss": np.mean(losses),
            "class_accuracy": self.class_wise_accuracy(y_true_total, y_pred_total),
            "y_true": y_true_total,
            "y_pred": y_pred_total,
        }
        return metrics

    def evaluation(self, model, datasets, tokenizer, batch_size, max_new_token):
        """
        evaluation at the end of the intire process
        """
        print("\n[Evaluation] Running evaluation loop...")
        start = default_timer()

        metrics = self.compute_metrics_on_dataset(
            model=model,
            dataset=datasets,
            tokenizer=tokenizer,
            max_new_tokens=max_new_token,
            batch_size=batch_size,
        )

        end = default_timer()

        print(f"[Evaluation] Accuracy: {metrics['accuracy']:.4f}")
        print(f"[Evaluation] Loss: {metrics['loss']:.4f}")
        print(f"[Evaluation] Time taken: {end - start:.2f}s")

        for label, acc in metrics["class_accuracy"].items():
            print(f"[Evaluation] Accuracy for class {label}: {acc:.4f}")

    def extract_label_from_response(self, list_strings):
        """
        use re to extract the label from the list of response or answer
        """
        # Regular expression to match "Y =" followed by any number (integer or float)
        list_match = [re.search(r"Y\s*=\s*(-?\d+(\.\d+)?)", i) for i in list_strings]
        list_y = [i.group() if i else "Y = 99" for i in list_match]

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


class MetricsCallback(TrainerCallback):
    def __init__(
        self, llm_instance: LLM, max_new_tokens, dataset, mode="train", trial=None
    ):
        self.llm = llm_instance
        self.max_new_tokens = max_new_tokens
        self.dataset = dataset
        self.mode = mode
        self.last_accuracy = None
        self.last_loss = None
        self.trial = trial

    def on_epoch_end(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        print(f"\n[MetricsCallback] Epoch {state.epoch} ended. Computing metrics...")

        # load tokenizer
        tokenizer = self.llm.load_tokenizer()

        # load model
        model = kwargs["model"]
        batch_size = args.per_device_train_batch_size * 6

        # get the metrics
        metrics = self.llm.compute_metrics_on_dataset(
            model=model,
            dataset=self.dataset,
            tokenizer=tokenizer,
            max_new_tokens=self.max_new_tokens,
            batch_size=batch_size,
        )

        # report the metrics
        self.last_accuracy = metrics["accuracy"]
        self.last_loss = metrics["loss"]

        print(f"[MetricsCallback] Accuracy: {self.last_accuracy:.4f}")
        print(f"[MetricsCallback] Loss: {self.last_loss:.4f}")

        if wandb.run:
            wandb.log(
                {
                    f"{self.mode}_custom_accuracy": self.last_accuracy,
                    f"{self.mode}_custom_loss": self.last_loss,
                    "epoch": state.epoch,
                },
                step=state.global_step,
            )

        for label, acc in metrics["class_accuracy"].items():
            print(f"[MetricsCallback] Class {label} accuracy: {acc:.4f}")
            if wandb.run:
                wandb.log(
                    {f"class_{label}_custom_accuracy": acc, "epoch": state.epoch},
                    step=state.global_step,
                )

        if self.trial is not None:
            self.trial.report(self.last_loss, step=state.epoch)
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()


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
    downsampling_n_instances = 300
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
