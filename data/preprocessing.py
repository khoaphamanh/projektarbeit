import os
from analysis import DataAnalysis
from sklearn.model_selection import train_test_split


class DataPreprocessing(DataAnalysis):
    def __init__(self, data_name):
        super().__init__(data_name)

    def load_data_raw_csv(self, extracted_label=None, extracted_feature = None):
        """
        load raw data as dataframe
        """
        # load dict analysis
        data_analysis_dict = self.analysis(extracted_label=extracted_label)

        #load all
        name_csv_files = data_analysis_dict.keys()
        for i in name_csv_files:
            data = 


if __name__ == "__main__":

    data_name = "HST"
    hst = DataPreprocessing(data_name)
    hst_data_dict = hst.load_data_raw_csv()
    # data_name = "TEP"
    # tep = DataAnalysis(data_name)
    # extracted_label = [0, 1, 4, 5]
    # tep_data_dict = tep.analysis(extracted_label=extracted_label, print_out=True)
    # print("tep_data_dict:", tep_data_dict)
