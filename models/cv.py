from llm import LLM
import optuna
import os
from sklearn.model_selection import StratifiedKFold
import numpy as np


class CrossValidation(LLM):
    def __init__(self, name_data, seed):
        super().__init__(name_data, seed)

    def cv(
        self,
        n_splits,
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
        save=False,
        quantized=False,
        batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=0.0001,
        weight_decay=0.001,
        max_new_tokens=5,
        save_model=False,
        epochs=10,
    ):
        """
        cross validation
        """
        train_val_datasets, test_datasets = self.load_data_llm(
            extracted_label=extracted_label,
            normalize=normalize,
            downsampling_n_instances=downsampling_n_instances,
            downsampling_n_instances_train=downsampling_n_instances_train,
            downsampling_n_instances_test=downsampling_n_instances_test,
            name_feature=name_feature,
            save_data=save,
        )

        # use stratifiedKFold
        answer = np.array(train_val_datasets["answer"])
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(answer)), answer)
        ):

            # indexing the split indices
            train_datasets = train_val_datasets.select(train_idx)
            val_datasets = train_val_datasets.select(val_idx)


# run this script
if __name__ == "__main__":

    seed = 1998

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
    weight_decay = 0.001
    max_new_tokens = 5
    save_model = False
    epochs = 150

    cv_run = CrossValidation(
        seed=seed,
        name_data=name_data,
    )

    cv_run.cv(
        n_splits=5,
        downsampling_n_instances=downsampling_n_instances,
    )
