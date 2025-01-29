import os
from analysis import DataAnalysis
from sklearn.model_selection import train_test_split
from datasets import Dataset


class DataPreprocessing(DataAnalysis):
    def __init__(self, name_data):
        super().__init__(name_data)

    def create_text_dataset(self, extracted_label=None, normalize=False):
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
            data_df = data_dict["df"]
            # print("data_df shape:", data_df.shape)

            # eliminate the column that has only 1
            data_df = data_df.loc[:, data_df.nunique() > 1]
            # print("data_df shape:", data_df.shape)

            # name of the feature
            feature_name = data_dict["feature_name"]
            # print("feature_name:", feature_name)

    def convert_df_to_text(self):
        pass

    def create_prompt(self):
        pass

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

    data_name = "HST"
    hst = DataPreprocessing(data_name)
    hst_data_dict = hst.create_text_dataset()
    # data_name = "TEP"
    # tep = DataAnalysis(data_name)
    # extracted_label = [0, 1, 4, 5]
    # tep_data_dict = tep.analysis(extracted_label=extracted_label, print_out=True)
    # print("tep_data_dict:", tep_data_dict)

    hst.system_behavior()
