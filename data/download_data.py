import requests
import zipfile
import os
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from tqdm import tqdm

# name of the data
name_hst = "HST.csv"
name_tep_dir = "TEP"

# name tep for each part of the data
tep_list = [
    "test_normal.csv",
    "train_normal.csv",
    "test_anomaly.csv",
    "train_anomaly.csv",
]


# function to donwload directory as .zip file, und zip and save it as .csv from RData
def download_and_extract_zip(url, output_path, tep_list=None):

    # download zip directory
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

    # list all files in folder TEP
    r_data_path = sorted(
        [os.path.join(output_path, i) for i in os.listdir(output_path) if "." in i]
    )

    for i, r_data in enumerate(tqdm(r_data_path, desc="Download data TEP")):

        # Enable automatic conversion between R and pandas data frames
        pandas2ri.activate()

        # Load the .RData file
        robjects.r["load"](r_data)

        # List all objects in the R environment to find the data frame name
        object_names = list(robjects.r.objects())

        # Replace 'df' with the correct object name from object_names list if it is different
        df_name = object_names[
            i
        ]  # Use the first object name or replace with the correct one if you know it
        df = robjects.r[df_name]

        # Convert to a pandas DataFrame
        df = pandas2ri.rpy2py(df)

        # Save the pandas DataFrame to CSV
        tep_data_csv_path = tep_list[i]
        df.to_csv(tep_data_csv_path, index=False)

        print(f"CSV file downloaded successfully: {tep_data_csv_path}")


# function to download .csv file given url
def download_file(url, output_path):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful

        with open(output_path, "wb") as file:
            file.write(response.content)

        print(f"CSV file downloaded successfully: {output_path}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download CSV file: {e}")


def download_data_convert_csv(name_data, url=None, tep_list=None):

    # Absolute path to current directory
    data_dir_path = os.path.dirname(os.path.abspath(__file__))

    # Full path to the data (csv file or directory)
    name_data_path = os.path.join(data_dir_path, name_data)

    # check data tep in .csv
    if tep_list is not None:
        tep_list = [os.path.join(data_dir_path, name_data_path, i) for i in tep_list]
        tep_list_check = [os.path.exists(i) for i in tep_list]

    # Check if the path exists
    if not os.path.exists(name_data_path):

        # If it doesn't exist, determine whether it's a file or a directory
        if name_data.endswith(".csv"):
            # If it's a CSV, download it
            download_file(url=url, output_path=name_data_path)
        else:
            # If it's not a CSV, assume it's a ZIP file to be extracted as a directory
            download_and_extract_zip(
                url=url, output_path=name_data_path, tep_list=tep_list
            )

    elif (
        os.path.exists(name_data_path)
        and tep_list is not None
        and not all(tep_list_check)
    ):
        download_and_extract_zip(url=url, output_path=name_data_path, tep_list=tep_list)

    else:
        if name_data.endswith(".csv"):
            print(f"{name_data_path} is already downloaded")

        elif tep_list is not None:
            for i in tep_list:
                print(f"{i} is already downloaded")
    print()


# link hst (direct download from seafile)
url_hst = "https://seafile.cloud.uni-hannover.de/f/23780066a22244899c94/?dl=1"
download_data_convert_csv(
    name_data=name_hst,
    url=url_hst,
)

# link TEP (direct download from seafile)
url_tep = "https://seafile.cloud.uni-hannover.de/f/98107a4c284f481eb6c0/?dl=1"
download_data_convert_csv(name_data=name_tep_dir, url=url_tep, tep_list=tep_list)
