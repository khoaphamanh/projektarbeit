import os
import pandas as pd
from download_data import *


# class to analysis data
class DataAnalysis:
    def __init__(self, data_name):
        self.data_name = data_name
        self.data_dir_path = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(self.data_dir_path, self.data_name)

    def check_data(self):
        # check if data name a file (True) or a directory (False)
        if os.path.isfile(self.data_name):
            return True
        return False

    def analysis(self):
        # analysis data based on given data name
        check = self.check_data()
        if check is True:
            1

    def csv_analysis(self, print_out=False):

        # dict analysis
        dict_analysis = {}
        dict_analysis["data_name"] = data_name

        # read the csv files given, in this case is HSV
        df = pd.read_csv(self.data_name)

        # Separate features and target (last column as target)
        features = df.iloc[:, :-1]
        target = df.iloc[:, -1]

        # calculate the number of unique label
        dict_analysis["num_unique_label"] = target.nunique()

        # Initialize lists for categorical and continuous features
        categorical_features = []
        continuous_features = []
        num_unique_categorical_features = []

        # Identify categorical and continuous features based on data types, int is categorical, float continious
        for column in features.columns:

            if pd.api.types.is_integer_dtype(features[column]):
                categorical_features.append(column)
                num_unique_categorical_features.append(features[column].nunique())

            elif pd.api.types.is_float_dtype(features[column]):
                continuous_features.append(column)

        dict_analysis["continuous_features"] = continuous_features
        dict_analysis["num_continuous_features"] = len(continuous_features)
        dict_analysis["categorical_features"] = categorical_features
        dict_analysis["num_categorical_features"] = len(categorical_features)
        dict_analysis["num_unique_categorical_features"] = (
            num_unique_categorical_features
        )

        # Calculate the number of features and instances
        num_features = len(features.columns)
        num_instances = len(df)

        dict_analysis["num_features"] = num_features
        dict_analysis["num_instances"] = num_instances

        # print out the analysis
        if print_out:
            print()
            for k, v in dict_analysis.items():
                print(k, v)


if __name__ == "__main__":
    data_name = "HST.csv"
    hst = DataAnalysis(data_name)
    data_name = hst.data_name
    print("data_name:", data_name)
    data_dir_path = hst.data_dir_path
    print("data_dir_path:", data_dir_path)
    hst.csv_analysis(print_out=True)
