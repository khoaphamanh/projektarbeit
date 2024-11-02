import os
import pandas as pd
from download_data import *


# class to analysis data
class DataAnalysis:
    def __init__(self, data_name):
        self.data_name = data_name
        self.data_dir_path = os.path.dirname(os.path.abspath(__file__))
        # print("self.data_dir_path", self.data_dir_path)
        self.data_path = os.path.join(self.data_dir_path, self.data_name)
        # print("self.data_path", self.data_path)

    def check_data(self):
        # check if data name a file (True) or a directory (False)
        if os.path.isfile(self.data_name):
            return True
        return False

    def analysis(self, print_out=False):
        # analysis data based on given data name (check if HST or TEP)
        check = self.check_data()
        if check is True:
            self.csv_analysis(print_out=print_out)
        else:
            tep_data = sorted(i for i in os.listdir(self.data_path) if ".csv" in i)
            for tep in tep_data:
                self.csv_analysis(csv_name=tep, print_out=True)

    def csv_analysis(self, csv_name=None, print_out=False):

        # dict analysis
        dict_analysis = {}

        # read the csv files given, in this case is HSV
        data_path = (
            self.data_path
            if csv_name is None
            else os.path.join(self.data_path, csv_name)
        )
        print("data_path:", data_path)
        df = pd.read_csv(data_path)

        # name of the data csv
        if csv_name is None:
            csv_name = self.data_name

        dict_analysis["data_name_csv"] = csv_name

        # Separate features and target (last column as target)
        if "HST" in self.data_name:
            features = df.iloc[:, :-1]
            target = df.iloc[:, -1]
        else:
            features = df.iloc[:, 1:]
            target = df.iloc[:, 0]

        # calculate the number of unique label
        dict_analysis["num_unique_label"] = target.nunique()

        # Initialize lists for categorical and continuous features
        features_name = []
        categorical_features = []
        continuous_features = []
        num_unique_categorical_features = []

        # Identify categorical and continuous features based on data types, int is categorical, float continious
        for column in features.columns:
            features_name.append(column)
            if pd.api.types.is_integer_dtype(features[column]):
                categorical_features.append(column)
                num_unique_categorical_features.append(features[column].nunique())

            elif pd.api.types.is_float_dtype(features[column]):
                continuous_features.append(column)

        dict_analysis["features_name"] = features_name
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
                print(k, "", v)

        return dict_analysis


if __name__ == "__main__":

    data_name = "HST.csv"
    hst = DataAnalysis(data_name)
    data_name = hst.data_name
    hst.analysis(print_out=True)

    data_name = "TEP"
    tep = DataAnalysis(data_name)
    data_name = hst.data_name
    tep.analysis(print_out=True)
