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
        self, extracted_label=None, normalize=False, downsampling_n_instances=None
    ):
        """
        load raw data as dataframe
        """
        # system behavior
        system_behavior = self.system_behavior(normalize=normalize)
        # load dict analysis
        data_analysis_dict = self.analysis(extracted_label=extracted_label)

        # load all data as data frame
        name_csv_files = data_analysis_dict.keys()
        for data_csv_name, data_dict in data_analysis_dict.items():

            # data as dataframe
            X_df = data_dict["X_df"]
            # print("X_df shape:", X_df.shape)
            y_df = data_dict["y_df"]

            # stratify downsampling
            if type(downsampling_n_instances) is int:
                X_df, y_df = self.stratified_downsampling(
                    X_df=X_df,
                    y_df=y_df,
                    downsampling_n_instances=downsampling_n_instances,
                )

            # eliminate the column that has only 1
            X_df = X_df.loc[:, X_df.nunique() > 1]

            # name of the feature
            name_feature = X_df.columns

            # unique labels
            # unique_labels =
            # split train and test
            X_df_train, X_df_test, y_df_train, y_df_test = (
                self.train_test_split_dataset(X_df, y_df)
            )

            # normalize
            if normalize:
                X_df_train, X_df_test = self.normalization(
                    X_df_train=X_df_train, X_df_test=X_df_test
                )

            # convert to array
            X_array_train, X_array_test, y_array_train, y_array_test = (
                self.convert_to_array(X_df_train, X_df_test, y_df_train, y_df_test)
            )
            # print("feature_name:", feature_name)

            self.convert_df_to_text(
                X_array=X_array_train, y_array=y_array_train, name_feature=name_feature
            )

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

    def train_test_split_dataset(self, X_df, y_df):
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

    def normalization(self, X_df_train, X_df_test):
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

    def convert_df_to_text(self, X_array, y_array, name_feature):
        """
        convert array to text given the X, y and feature name
        """
        # data text as list of all instances in X
        data_text_list_all_instances = []

        # unique labels of y
        unique_labels_string = " or ".join(map(str, np.unique(y_array).tolist()))

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
            {"text": self.create_prompt(d["question"], d["answer"])}
            for d in data_text_list_all_instances
        ]
        return formatted_data

    def create_prompt(self, question, answer):
        """create prompt using the syntax of llama-2"""
        return f"<s>[INST] <<SYS>> {self.system_behavior()} <</SYS>> {question} [/INST] {answer} </s>"

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
    hst = DataPreprocessing(data_name, seed=seed)
    hst_data_dict = hst.create_text_dataset(
        downsampling_n_instances=300, normalize=True
    )
    # data_name = "TEP"
    # tep = DataAnalysis(data_name)
    # extracted_label = [0, 1, 4, 5]
    # tep_data_dict = tep.analysis(extracted_label=extracted_label, print_out=True)
    # print("tep_data_dict:", tep_data_dict)

    # hst.system_behavior()
