import os
import pandas as pd


class DataAnalysis:
    def __init__(self, data_name):
        self.data_name = data_name
        self.data_dir_path = os.path.abspath(__file__)
        self.a = 1

    def check_data(self):
        # check if data name a file (True) or a directory (False)
        if os.path.isfile(self.data_name):
            return True
        return False

    def analysis(self, index=None):
        # analysis data based on given data name
        check = self.check_data()
        if check is True:
            data = pd.read_csv(self.data_name)

    def read_csv_analysis (data_name_csv):
        #read the csv files given 
        
if __name__ == "__main__":
    data_name = "a"
    hst = DataAnalysis(data_name)
    data_name = hst.data_name
    print("data_name:", data_name)
    data_dir_path = hst.data_dir_path
    print("data_dir_path:", data_dir_path)
