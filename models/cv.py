from llm import LLM
import optuna
import os
from sklearn.model_selection import StratifiedKFold
import numpy as np
import wandb
import argparse
import gc
import torch

# Set environment variable before torch is imported
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CrossValidation(LLM):
    def __init__(self, name_data, seed):
        super().__init__(name_data, seed)

    def cv(
        self,
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
        hpo=False,
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
            save_data=save_data,
        )

        # use stratifiedKFold
        answer = np.array(train_val_datasets["answer"])
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)

        # list to calculate performance each trial
        list_loss_val = []
        list_accuracy_val = []

        for index_split, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(answer)), answer)
        ):

            # indexing the split indices
            train_datasets = train_val_datasets.select(train_idx)
            val_datasets = train_val_datasets.select(val_idx)

            # run classification
            metrics_val = self.run_classification(
                train_datasets=train_datasets,
                test_datasets=val_datasets,
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
                trial=trial,
                n_splits=n_splits,
                index_split=index_split,
                hpo=hpo,
            )

            # append to lists
            loss = metrics_val["loss"]
            accuracy = metrics_val["accuracy"]
            list_loss_val.append(loss)
            list_accuracy_val.append(accuracy)

            # print out the metrics
            if trial is not None:
                print(
                    f"Trial {trial.number} - Split {index_split} - Loss val: {loss:.4f} - Accuracy val: {accuracy:.4f}"
                )

        # calcualte mean
        loss_mean_val = np.mean(list_loss_val)
        accuracy_mean_val = np.mean(list_accuracy_val)

        return loss_mean_val, accuracy_mean_val


def objective(trial: optuna.trial.Trial):

    # suggest the parameters with search space
    lora_r = trial.suggest_int("lora_r", low=4, high=32, step=1)
    lora_alpha = trial.suggest_int("lora_alpha", low=4, high=128, step=1)
    lora_dropout = trial.suggest_float("lora_dropout", low=0.05, high=1, step=0.01)
    normalize = trial.suggest_categorical("normalize", choices=[True, False])
    learning_rate = trial.suggest_float("learning_rate", low=1e-5, high=1e-2, step=1e-5)

    # conditional hyperparmameters
    name_feature = (
        trial.suggest_categorical("name_feature", choices=[True, False])
        if name_data == "HST"
        else False
    )

    # init cv
    cv_run = CrossValidation(
        seed=seed,
        name_data=name_data,
    )

    # run cross validation
    loss_val, accuracy_val = cv_run.cv(
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
        trial=trial,
        n_splits=n_splits,
        hpo=hpo,
    )

    return loss_val


# run this script
if __name__ == "__main__":

    # hpo argument argparse
    parser = argparse.ArgumentParser(description="Choose data name, to perform HPO")
    parser.add_argument(
        "-d",
        "--data_name",
        type=str,
        help="choose data name between 'HST' or 'TEP', default is 'HST'",
        choices=["HST", "TEP"],
    )

    args = parser.parse_args()

    # set data name
    if args.data_name == "HST":
        name_data = "HST"
    elif args.data_name == "TEP":
        name_data = "TEP"
    print("name_data:", name_data)

    # fix hyperparameters for hpo
    seed = 1998
    n_trials = 100
    extracted_label = None if name_data == "HST" else [0, 1, 4, 5]
    downsampling_n_instances = 30 if name_data == "HST" else None
    downsampling_n_instances_train = 400 if name_data == "TEP" else None
    downsampling_n_instances_test = 160 if name_data == "TEP" else None
    quantized = True
    batch_size = 1
    gradient_accumulation_steps = 1
    max_new_tokens = 5
    save_model = False
    epochs = 3
    n_splits = 5
    seed = 1998
    save_data = False
    hpo = True

    # path to sqlite database
    path_models_directory = os.path.dirname(os.path.abspath(__file__))
    path_hpo_directory = os.path.join(path_models_directory, "hpo")
    os.makedirs(path_hpo_directory, exist_ok=True)
    db_hpo = "hpo.db"
    path_db_hpo = os.path.join(path_hpo_directory, db_hpo)
    db_hpo_sqlite = f"sqlite:///{path_db_hpo}"

    # path csv optuna
    path_csv_hpo = os.path.join(path_hpo_directory, f"hpo_{name_data}.csv")

    # sampler and pruner in optuna
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner()
    study_name = f"hpo_{name_data}"

    # create or load a exist study
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=db_hpo_sqlite,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    # export db optuna to csv
    df_trials = study.trials_dataframe()
    df_trials.to_csv(path_csv_hpo, index=False)

    # run trials if current trials < n_trials
    if len(study.trials) < n_trials:

        # check if the last trial is failed
        if len(study.trials) > 0 and study.trials[-1].state in [
            optuna.trial.TrialState.FAIL,
            optuna.trial.TrialState.RUNNING,
        ]:
            failed_trial_params = study.trials[-1].params
            study.enqueue_trial(failed_trial_params)

        # run trials
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True,
        )

    else:
        pruned_trials = study.get_trials(
            deepcopy=False, states=[optuna.trial.TrialState.PRUNED]
        )
        complete_trials = study.get_trials(
            deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
        )
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        best_trial_score = study.best_trial
        best_trial_params = best_trial_score.params

        print("  Value: ", best_trial_score.value)

        print("  Params: ")
        for key, value in best_trial_params.items():
            print("    {}: {}".format(key, value))

        # apply final best trial
        cv = CrossValidation(
            seed=seed,
            name_data=name_data,
        )

        metrics_test = cv.classification(
            lora_r=best_trial_params["lora_r"],
            lora_alpha=best_trial_params["lora_alpha"],
            lora_dropout=best_trial_params["lora_dropout"],
            extracted_label=extracted_label,
            normalize=best_trial_params["normalize"],
            downsampling_n_instances=downsampling_n_instances,
            downsampling_n_instances_train=downsampling_n_instances_train,
            downsampling_n_instances_test=downsampling_n_instances_test,
            name_feature=(
                best_trial_params["name_feature"] if name_data == "HST" else False
            ),
            save_data=save_data,
            quantized=quantized,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=best_trial_params["learning_rate"],
            max_new_tokens=max_new_tokens,
            save_model=True,
            epochs=epochs,
            trial=None,
            n_splits=None,
            index_trial=None,
            hpo=False,
        )

        # report the final test metrics
        for key, value in metrics_test.items():
            print(f"{key}: {value}")
