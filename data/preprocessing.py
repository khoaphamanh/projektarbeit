import os
from analysis import DataAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datasets import Dataset
import pandas as pd
import numpy as np


class DataPreprocessing(DataAnalysis):
    def __init__(self, name_data, seed=1998):
        super().__init__(name_data)
        self.seed = seed
        self.train_size = 0.8

    def create_text_dataset(
        self,
        extracted_label=None,
        normalize=False,
        downsampling_n_instances=None,
        downsampling_n_instances_train=None,
        downsampling_n_instances_test=None,
        name_feature=False,
    ):
        """
        load raw data as dataframe
        """
        # load dict analysis
        data_analysis_dict = self.analysis(extracted_label=extracted_label)

        # load data not splited in HST data
        if self.name_data == "HST":
            # data as dataframe
            data_dict = data_analysis_dict[f"{self.name_data}.csv"]
            X_df = data_dict["X_df"]
            y_df = data_dict["y_df"]

            # split the dataset HST
            X_df_train, X_df_test, y_df_train, y_df_test = (
                self.preprocessing_before_train_test_split(
                    X_df=X_df,
                    y_df=y_df,
                    downsampling_n_instances=downsampling_n_instances,
                )
            )

            # preprcessing after train test split HST
            train_datasets, test_datasets = self.preprocessing_after_train_test_split(
                X_df_train=X_df_train,
                X_df_test=X_df_test,
                y_df_train=y_df_train,
                y_df_test=y_df_test,
                normalize=normalize,
                name_feature=name_feature,
            )

        # load data plited in TEP data
        elif self.name_data == "TEP":
            X_df_train = []
            y_df_train = []
            X_df_test = []
            y_df_test = []

            # loop through all data files that start with train and tests
            for name_file, data_dict in data_analysis_dict.items():
                X_df = data_dict["X_df"]
                y_df = data_dict["y_df"]
                if name_file.startwith("train"):
                    X_df_train.append(X_df)
                    y_df_train.append(y_df)
                else:
                    X_df_test.append(X_df)
                    y_df_test.append(y_df)

            # concat train and test dataset (normal and anomaly) of TEP
            X_df_train = pd.concat(X_df_train, axis=0)
            print("X_df_train shape:", X_df_train.shape)
            y_df_train = pd.concat(y_df_train, axis=0)
            print("y_df_train shape:", y_df_train.shape)
            X_df_test = pd.concat(X_df_test, axis=0)
            print("X_df_test shape:", X_df_test.shape)
            y_df_test = pd.concat(y_df_test, axis=0)
            print("y_df_test shape:", y_df_test.shape)

            # preprcessing after train test split TEP
            train_datasets, test_datasets = self.preprocessing_after_train_test_split(
                X_df_train=X_df_train,
                X_df_test=X_df_test,
                y_df_train=y_df_train,
                y_df_test=y_df_test,
                normalize=normalize,
                name_feature=name_feature,
            )

        for i in test_datasets:
            print(i)

    def preprocessing_before_train_test_split(
        self,
        X_df: pd.DataFrame,
        y_df: pd.DataFrame,
        downsampling_n_instances=None,
    ):
        # stratify downsampling
        if type(downsampling_n_instances) is int:
            X_df, y_df = self.stratified_downsampling(
                X_df=X_df,
                y_df=y_df,
                downsampling_n_instances=downsampling_n_instances,
            )

        # eliminate the column that has only 1
        X_df = X_df.loc[:, X_df.nunique() > 1]

        # split train and test
        X_df_train, X_df_test, y_df_train, y_df_test = self.train_test_split_dataset(
            X_df, y_df
        )

        return X_df_train, X_df_test, y_df_train, y_df_test

    def preprocessing_after_train_test_split(
        self,
        X_df_train,
        X_df_test,
        y_df_train,
        y_df_test,
        downsampling_n_instances_train=None,
        downsampling_n_instances_test=None,
        normalize=False,
        name_feature=False,
    ):
        """
        preprocessing after train test split
        """
        # system behavior
        system_behavior = self.system_behavior(normalize=normalize)

        # stratify downsampling
        if (
            type(downsampling_n_instances_train) is int
            and type(downsampling_n_instances_test) is int
        ):
            X_df_train, y_df_train = self.stratified_downsampling(
                X_df=X_df_train,
                y_df=y_df_train,
                downsampling_n_instances=downsampling_n_instances_train,
            )

            X_df_test, y_df_test = self.stratified_downsampling(
                X_df=X_df_test,
                y_df=y_df_test,
                downsampling_n_instances=downsampling_n_instances_test,
            )

        # convert to text and use prompt
        if name_feature:
            name_feature = X_df_train.columns

        # normalize
        if normalize:
            X_df_train, X_df_test = self.normalization(
                X_df_train=X_df_train, X_df_test=X_df_test
            )

        # convert to array
        X_array_train, X_array_test, y_array_train, y_array_test = (
            self.convert_to_array(X_df_train, X_df_test, y_df_train, y_df_test)
        )

        # convert to text and create prompt
        train_datasets = self.convert_df_to_text(
            X_array=X_array_train,
            y_array=y_array_train,
            name_feature=name_feature,
            system_behavior=system_behavior,
        )
        test_datasets = self.convert_df_to_text(
            X_array=X_array_test,
            y_array=y_array_test,
            name_feature=name_feature,
            system_behavior=system_behavior,
        )

        return train_datasets, test_datasets

    def stratified_downsampling(
        self, X_df: pd.DataFrame, y_df: pd.DataFrame, downsampling_n_instances: int
    ):
        """
        down sampling the dataset with same ratio of unique labels
        """
        # check number of unique labels
        unique_labels = y_df.unique()

        # number of instances each label:
        n_instances_each_labels = downsampling_n_instances // len(unique_labels)

        # downsampling the data
        X_df_downsampling = []
        y_df_downsampling = []

        for l in unique_labels:
            # sampling, each instance will be different
            X_df_l = X_df[y_df == l].sample(
                n=n_instances_each_labels, random_state=self.seed
            )
            y_df_l = y_df[y_df == l].sample(
                n=n_instances_each_labels, random_state=self.seed
            )
            # append to list
            X_df_downsampling.append(X_df_l)
            y_df_downsampling.append(y_df_l)

        X_df_downsampling = pd.concat(X_df_downsampling, axis=0)
        y_df_downsampling = pd.concat(y_df_downsampling, axis=0)

        return X_df_downsampling, y_df_downsampling

    def train_test_split_dataset(self, X_df: pd.DataFrame, y_df: pd.DataFrame):
        """
        train test split with stratify y
        """
        X_df_train, X_df_test, y_df_train, y_df_test = train_test_split(
            X_df,
            y_df,
            train_size=self.train_size,
            random_state=self.seed,
            stratify=y_df,
        )
        return X_df_train, X_df_test, y_df_train, y_df_test

    def normalization(self, X_df_train: pd.DataFrame, X_df_test: pd.DataFrame):
        """
        Normalize data with Min-Max-Scaler
        """
        # scaler as min max scaler
        scaler = MinMaxScaler()

        # fit the scaler
        scaler.fit(X_df_train)

        # transform the data
        X_df_train_scaled = scaler.transform(X_df_train)
        X_df_test_scaled = scaler.transform(X_df_test)

        return X_df_train_scaled, X_df_test_scaled

    def convert_to_array(self, *args):
        """
        convert data frame to array
        """
        return (np.array(arg) for arg in args)

    def convert_df_to_text(self, X_array, y_array, name_feature, system_behavior):
        """
        convert array to text given the X, y and feature name
        """
        # data text as list of all instances in X
        data_text_list_all_instances = []

        # unique labels of y as text
        unique_labels_string = " or ".join(map(str, np.unique(y_array).tolist()))

        # check if name_feature is given
        if name_feature is None or name_feature is False:
            name_feature = list(range(len(y_array)))

        # loop through all samples from X and y
        for i in range(len(X_array)):
            X_instance = X_array[i]
            y_instance = y_array[i]
            question_one_instance_from_array = []

            # loop through all feature value from one instance from X
            for idx, value in enumerate(X_instance):

                # create text instance for all feature values
                question_one_instance_from_array.append(
                    "feature {} = {}".format(name_feature[idx], value)
                )

            # create text instance for all feature values from one instance from X
            question_one_instance_from_array = ", ".join(
                question_one_instance_from_array
            )
            # create text question instance for one instance from X
            question_one_instance = (
                "Tell me if the value of Y is {}. If {}. What should be Y?".format(
                    unique_labels_string, question_one_instance_from_array
                )
            )
            # create text answer instance for one instance from X
            answer_one_instance = "Y = {}".format(y_instance)

            # append to data text list of all instance
            data_text_list_all_instances.append(
                {"question": question_one_instance, "answer": answer_one_instance}
            )

        # format with create_prompt function
        formatted_data = [
            {
                "text": self.create_prompt(
                    question=d["question"],
                    answer=d["answer"],
                    system_behavior=system_behavior,
                )
            }
            for d in data_text_list_all_instances
        ]

        return Dataset.from_list(formatted_data)

    def create_prompt(self, question, answer, system_behavior):
        """create prompt using the syntax of llama-2"""
        return f"<s>[INST] <<SYS>> {system_behavior} <</SYS>> {question} [/INST] {answer} </s>"

    def system_behavior(self, normalize=False):
        """
        system behavior of the model, in this case is instruction in the paper
        """
        # full name of the dataset
        if self.name_data == "HST":
            name_full_data = "High-speed Train Braking System Fault Detection Dataset"
        elif self.name_data == "TEP":
            name_full_data = "Tennessee Eastman Process Simulation Dataset"

        # text if apply normalize
        if normalize:
            text_normalize = "The following is the data after normalization, with the numerical values ranging between 0 and 1."
        else:
            text_normalize = ""

        system_behavior = (
            f"You are an expert in fault diagnosis of chemical plants operation. "
            f"You master the reaction process and control structures in the {name_full_data}. "
            f"You are capable of accurately determining the plant process state based on given variables and their values. "
            f"Below is a sample of the {name_full_data} monitoring.{text_normalize} "
            f"Please determine the process state based on your knowledge "
            f"of the {name_full_data} and the given data."
        )
        return system_behavior


if __name__ == "__main__":

    seed = 1998
    data_name = "HST"
    # hst = DataPreprocessing(data_name, seed=seed)
    # hst_data_dict = hst.create_text_dataset(
    #     downsampling_n_instances=300, normalize=True, name_feature=True
    # )

    data_name = "TEP"

    extracted_label = [0, 1, 4, 5]
    tep = DataPreprocessing(data_name, seed=seed)
    tep.create_text_dataset(
        extracted_label=extracted_label,
        normalize=True,
        downsampling_n_instances_train=400,
        downsampling_n_instances_test=160,
        name_feature=True,
    )
