import requests
import zipfile
import os

#data directory path
data_dir_path = os.path.abspath(__file__)
name_hst = "HST.csv"

def download_extract_zip (url,output_path):
    
    