import requests
import zipfile
import os
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import pandas as pd

# name of the data
name_hst = "HST.csv"
name_tep_dir = "TEP"

# name tep for each part of the data
name_tep = ["train_normal.csv", "train_anomaly", "test_normal", "test_anomaly"]


# function to download .csv file given url
def download_file(url, output_path):

    # download the file in this case is .csv file
    try:
        # Send a request to the URL
        response = requests.get(url)

        # Save the CSV file
        with open(output_path, "wb") as file:
            file.write(response.content)

        print(f"CSV file downloaded successfully: {output_path}")

    except requests.exceptions.RequestException as e:
        print(f"Failed to download CSV file: {e}")


# function to donwload directory as .zip file, und zip and save it as .csv from RData
def download_and_extract_zip(url, output_path):
    try:
        # Send a request to the URL
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check if the request was successful

        # Save the ZIP file temporarily
        temp_zip_path = "temp_download.zip"
        with open(temp_zip_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        # Extract the contents of the ZIP file
        with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
            zip_ref.extractall(output_path)

        print(f"ZIP directory downloaded and extracted successfully: {output_path}")

        # Remove the temporary ZIP file
        os.remove(temp_zip_path)

    except requests.exceptions.RequestException as e:
        print(f"Failed to download ZIP file: {e}")
    except zipfile.BadZipFile:
        print(f"Failed to extract: Not a valid ZIP file")


def download_data_convert_csv(name_data, url=None, name_tep_csv=None):

    # data directory path
    data_dir_path = os.path.dirname(os.path.abspath(__file__))

    # download data files and save it to data folder. HST data as .csv and TEP data as
    name_data_path = os.path.join(data_dir_path, name_data)

    # check if files or directory available given name_data_path
    if not os.path.exists(name_data_path):

        # check if a file .csv given name_data_path
        if not os.path.isfile(name_data_path):

            # download .csv file from cloud driver
            download_file(url=url, output_path=name_data_path)

        # check if a directory TEP given name_data_path
        elif not os.path.isdir(name_data_path):

            # download the TEP directory
            download_and_extract_zip(url=url, output_path=name_data_path)
            # #check if all .csv file tep available
            # check_csv_tep = [os.path.exists(i) for i in name_tep_csv]

            # if not all(check_csv_tep):

    else:
        print("{} is already downloaded".format(name_data_path))


if __name__ == "__main__":

    # link hst (direct download from seafile)
    url_hst = "https://seafile.cloud.uni-hannover.de/f/23780066a22244899c94/?dl=1"
    download_data_convert_csv(
        name_data=name_hst,
        url=url_hst,
    )

    # link TEP
    url_tep = "https://seafile.cloud.uni-hannover.de/f/98107a4c284f481eb6c0/?dl=1"
    download_data_convert_csv(
        name_data=name_tep_dir,
        url=url_tep,
    )
