import os
from analysis import DataAnalysis
from sklearn.model_selection import train_test_split
from datasets import Dataset


class DataPreprocessing(DataAnalysis):
    def __init__(self, data_name):
        super().__init__(data_name)

    def load_data_raw_csv(self, extracted_label=None, normalize=True):
        """
        load raw data as dataframe
        """
        # load dict analysis
        data_analysis_dict = self.analysis(extracted_label=extracted_label)

        # load all data as data frame
        name_csv_files = data_analysis_dict.keys()
        for data_csv_name, data_dict in data_analysis_dict.items():

            # data as dataframe
            data_df = data_dict["df"]
            print("data_df shape:", data_df.shape)

            # eliminate the column that has only 1
            data_df = data_df.loc[:, data_df.nunique() > 1]
            print("data_df shape:", data_df.shape)

            # name of the feature
            feature_name = data_dict["feature_name"]
            print("feature_name:", feature_name)

    def convert_df_to_text(self):
        pass

    def create_prompt(self):
        pass


if __name__ == "__main__":

    data_name = "HST"
    hst = DataPreprocessing(data_name)
    hst_data_dict = hst.load_data_raw_csv()
    # data_name = "TEP"
    # tep = DataAnalysis(data_name)
    # extracted_label = [0, 1, 4, 5]
    # tep_data_dict = tep.analysis(extracted_label=extracted_label, print_out=True)
    # print("tep_data_dict:", tep_data_dict)
