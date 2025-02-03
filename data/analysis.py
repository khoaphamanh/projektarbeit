import os
import pandas as pd


# class to analysis data
class DataAnalysis:
    def __init__(self, name_data):
        self.name_data = name_data
        self.path_data_directory = os.path.dirname(os.path.abspath(__file__))
        self.path_name_data_directory = os.path.join(
            self.path_data_directory, self.name_data
        )
        self.check_data()

    def check_data(self):
        """
        check if data available, if not download data
        """
        # check if data available
        if not os.path.exists(self.path_name_data_directory):
            import download_data

    def analysis(self, extracted_label=None, print_out=False):
        """
        analysis the data, return the dict of name data, csv_files in dât, dataframe and its informations
        """
        # analysis data based on given data name (check if HST or TEP)
        data_csv = sorted(
            i for i in os.listdir(self.path_name_data_directory) if ".csv" in i
        )
        data_analysis_dict = {}
        for file_csv in data_csv:
            df = self.csv_analysis(
                file_csv_name=file_csv,
                extracted_label=extracted_label,
                print_out=print_out,
            )
            data_analysis_dict[file_csv] = df

        return data_analysis_dict

    def csv_analysis(
        self,
        file_csv_name=None,
        extracted_label=None,
        print_out=False,
    ):
        """
        analysis of each csv file of the data
        """
        # dict analysis
        dict_analysis = {}

        # read the csv files given, in this case is HSV
        data_path = os.path.join(self.path_name_data_directory, file_csv_name)
        df = pd.read_csv(data_path)

        # name of the data csv
        dict_analysis["name_csv"] = file_csv_name

        # Separate features and target (last column as target)
        if "HST" in self.name_data:
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
        else:
            X = df.iloc[:, 1:]
            y = df.iloc[:, 0]

        # Calculate the number of features and instances
        dict_analysis["num_instances"] = len(X)
        dict_analysis["num_features"] = len(X.columns)

        # extract label
        if extracted_label is not None:
            index_extracted_label = y.isin(extracted_label)
            X = X[index_extracted_label]
            y = y[index_extracted_label]
            dict_analysis["num_instances_extracted"] = len(X)

            # available label
            available_lable = [
                label for label in extracted_label if label in y.tolist()
            ]
            dict_analysis["extracted_label"] = extracted_label
            dict_analysis["available_label"] = available_lable

        # calculate the number of unique label
        dict_analysis["num_unique_label"] = y.nunique()
        dict_analysis["unique_label"] = y.unique()
        num_instances_each_label = y.value_counts().tolist()
        dict_analysis["num_instances_each_label"] = num_instances_each_label

        # Initialize lists for categorical and continuous features
        features_name = []
        categorical_features = []
        continuous_features = []
        num_unique_categorical_features = []

        # Identify categorical and continuous features based on data types, int is categorical, float continious
        for column in X.columns:
            features_name.append(column)
            if pd.api.types.is_integer_dtype(X[column]):
                categorical_features.append(column)
                num_unique_categorical_features.append(X[column].nunique())

            elif pd.api.types.is_float_dtype(X[column]):
                continuous_features.append(column)

        dict_analysis["feature_name"] = features_name
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

        dict_analysis["X_df"] = X
        dict_analysis["y_df"] = y

        return dict_analysis


if __name__ == "__main__":

    # data_name = "HST"
    # hst = DataAnalysis(data_name)
    # hst_data_dict = hst.analysis(print_out=True)
    # print("hst_data_dict:", hst_data_dict)

    data_name = "TEP"
    tep = DataAnalysis(data_name)
    extracted_label = [0, 1, 4, 5]
    tep_data_dict = tep.analysis(extracted_label=extracted_label, print_out=True)
    print("tep_data_dict:", tep_data_dict)
