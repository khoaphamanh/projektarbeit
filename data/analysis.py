import os
import pandas as pd


# class to analysis data
class DataAnalysis:
    def __init__(self, data_name):
        self.data_name = data_name
        self.data_dir_path = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(self.data_dir_path, self.data_name)
        self.check_data()

    def check_data(self):
        # check if data available
        if not os.path.isfile(self.data_name):
            import download_data

    def analysis(self, extracted_label=None, print_out=False):
        # analysis data based on given data name (check if HST or TEP)
        data_csv = sorted(i for i in os.listdir(self.data_path) if ".csv" in i)
        for file_csv in data_csv:
            self.csv_analysis(
                csv_name=file_csv, extracted_label=extracted_label, print_out=print_out
            )

    def csv_analysis(self, csv_name=None, extracted_label=None, print_out=False):

        # dict analysis
        dict_analysis = {}

        # read the csv files given, in this case is HSV
        data_path = (
            self.data_path
            if csv_name is None
            else os.path.join(self.data_path, csv_name)
        )
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

        # Calculate the number of features and instances
        dict_analysis["num_instances"] = len(features)
        dict_analysis["num_features"] = len(features.columns)

        # extract label
        if extracted_label is not None:
            index_extracted_label = target.isin(extracted_label)
            features = features[index_extracted_label]
            target = target[index_extracted_label]
            dict_analysis["num_instances_extracted"] = len(features)

            # available label
            available_lable = [
                label for label in extracted_label if label in target.tolist()
            ]
            dict_analysis["available_label"] = available_lable

        # calculate the number of unique label
        dict_analysis["num_unique_label"] = target.nunique()
        dict_analysis["unique_label"] = target.unique()
        num_instances_each_label = target.value_counts().tolist()
        dict_analysis["num_instances_each_label"] = num_instances_each_label

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

        # print out the analysis
        if print_out:
            for k, v in dict_analysis.items():
                print(k, ":", v)
            print()

        return dict_analysis


if __name__ == "__main__":

    data_name = "HST"
    hst = DataAnalysis(data_name)
    hst.analysis(print_out=True)

    data_name = "TEP"
    tep = DataAnalysis(data_name)
    extracted_label = [0, 1, 4, 5]
    tep.analysis(extracted_label=extracted_label, print_out=True)
