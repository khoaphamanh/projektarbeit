from analysis import DataAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datasets import Dataset
import pandas as pd
import numpy as np
import os


class DataPreprocessing(DataAnalysis):
    def __init__(self, name_data, seed=1998):
        super().__init__(name_data)

        # default parameters
        self.seed = seed
        self.train_size = 0.8

    def load_data_llm_format(
        self,
        extracted_label=None,
        normalize=False,
        downsampling_n_instances=None,
        downsampling_n_instances_train=None,
        downsampling_n_instances_test=None,
        name_feature=False,
        save=False,
    ):
        """
        load a saved dataset, not use for HPO
        """
        # path of train and test dataset
        path_train_datasets = self.path_name_data_llm_format_directory(kind="train")
        path_test_datasets = self.path_name_data_llm_format_directory(kind="train")

        # load data if exsis
        if os.path.exists(path_train_datasets) and os.path.exists(path_test_datasets):
            train_datasets = Dataset.load_from_disk(path_train_datasets)
            test_datasets = Dataset.load_from_disk(path_test_datasets)

        else:
            train_datasets, test_datasets = self.create_text_dataset(
                extracted_label=extracted_label,
                normalize=normalize,
                downsampling_n_instances=downsampling_n_instances,
                downsampling_n_instances_train=downsampling_n_instances_train,
                downsampling_n_instances_test=downsampling_n_instances_test,
                name_feature=name_feature,
                save=save,
            )

        return train_datasets, test_datasets

    def create_text_dataset(
        self,
        extracted_label=None,
        normalize=False,
        downsampling_n_instances=None,
        downsampling_n_instances_train=None,
        downsampling_n_instances_test=None,
        name_feature=False,
        save=False,
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

                # sort the name
                if name_file.startswith("train"):
                    X_df_train.append(X_df)
                    y_df_train.append(y_df)
                else:
                    X_df_test.append(X_df)
                    y_df_test.append(y_df)

            # concat train and test dataset (normal and anomaly) of TEP
            X_df_train = pd.concat(X_df_train, axis=0)
            y_df_train = pd.concat(y_df_train, axis=0)
            X_df_test = pd.concat(X_df_test, axis=0)
            y_df_test = pd.concat(y_df_test, axis=0)

            # preprcessing after train test split TEP
            train_datasets, test_datasets = self.preprocessing_after_train_test_split(
                X_df_train=X_df_train,
                X_df_test=X_df_test,
                y_df_train=y_df_train,
                y_df_test=y_df_test,
                downsampling_n_instances_train=downsampling_n_instances_train,
                downsampling_n_instances_test=downsampling_n_instances_test,
                normalize=normalize,
                name_feature=name_feature,
            )

        # save the data llm format
        if save:
            train_datasets.save_to_disk(
                dataset_path=self.path_name_data_llm_format_directory(kind="train")
            )
            test_datasets.save_to_disk(
                dataset_path=self.path_name_data_llm_format_directory(kind="test")
            )

        return train_datasets, test_datasets

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

        # loop through each labels
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
                ),
                "question": self.create_prompt(question=d["question"]),
                "answer": d["answer"],
            }
            for d in data_text_list_all_instances
        ]

        return Dataset.from_list(formatted_data)

    def create_prompt(self, question, answer=None, system_behavior=None):
        """create prompt using the syntax of llama-2"""
        if answer is not None and system_behavior is not None:
            return f"<s>[INST] <<SYS>> {system_behavior} <</SYS>> {question} [/INST] {answer} </s>"
        else:
            return f"<s>[INST] {question} [/INST]"

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

    def path_name_data_llm_format_directory(self, kind):
        """
        create path of saved data in llm format
        """
        name_data_llm_format_directory = (
            f"{self.name_data}_{self.seed}_{kind}_llm_format"
        )
        path_name_data_llm_format_directory = os.path.join(
            self.path_name_data_directory, name_data_llm_format_directory
        )
        return path_name_data_llm_format_directory


if __name__ == "__main__":

    from timeit import default_timer

    start = default_timer()
    seed = 1998

    # hyperparameters
    normalize = True
    name_feature = True
    downsampling_n_instances = 300
    downsampling_n_instances_train = 400
    downsampling_n_instances_test = 160

    # load data
    data_name = "HST"
    hst = DataPreprocessing(data_name, seed=seed)
    hst_train, hst_test = hst.load_data_llm_format(
        downsampling_n_instances=300, normalize=True, name_feature=True, save=True
    )

    # for i in hst_test:
    #     print(i)

    data_name = "TEP"
    extracted_label = [0, 1, 4, 5]
    tep = DataPreprocessing(data_name, seed=seed)
    tep_train, tep_test = tep.load_data_llm_format(
        extracted_label=extracted_label,
        normalize=True,
        downsampling_n_instances_train=400,
        downsampling_n_instances_test=160,
        name_feature=True,
        save=True,
    )

    for i in tep_test:
        print(i)

    # import re
    # def extract_question_answer(example):
    #     """
    #     Extracts the question (inside [INST] ... [/INST]) and the answer (after [/INST])
    #     """
    #     pattern = r"\[INST\](.*?)\[/INST\](.*)"
    #     match = re.search(pattern, example["text"], re.DOTALL)

    #     if match:
    #         question = match.group(1).strip()  # Extract the text inside [INST]
    #         answer = match.group(2).strip()  # Extract text after [/INST]

    #         # Remove the <<SYS>> and <</SYS>> system instructions (optional)
    #         question = re.sub(
    #             r"<<SYS>>.*?<</SYS>>", "", question, flags=re.DOTALL
    #         ).strip()

    #         return {"question": question, "answer": answer}
    #     else:
    #         return {"question": None, "answer": None}

    # split_dataset = tep_test.map(extract_question_answer)

    # for i in split_dataset:
    #     print(i.keys())
    #     print(i)
    #     break

    end = default_timer()
    print(end - start)
