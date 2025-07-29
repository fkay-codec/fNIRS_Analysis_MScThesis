"""
This script extracts individual-subject GLM beta coefficients from seed-based connectivity (with seed as predictor in GLM)
into an Excel files that has the betas from all participants with the goal to perform a one-sample t-tests. 

Input: 
    - individual GLM results with beta coefficients
Output:
    - Single Excel file with all betas from all participants ready for group analysis.

Usage:
    1. Select the folder containing the Satori Individual GLM results (e.g., 'SBA_GLM_Oxy') 
    2. The script will itterate through the folder, select the excel files only, and extract the beta's into a single DF
        - it also extracts: channel names, seed name, participant ID
        - Example output DF with name- (...)_{seed_name}:
    | Channels  | Participant_1 | Participant_2 | ... | Participant_N |
    | S1D1      | 0.123         | 0.456         | ... |

Note: it was double checked that the order of the channels is the same and the betas are written correctly
"""

import os
import pandas as pd
from PySide6.QtWidgets import QApplication, QFileDialog
import sys

def extract_clean_name(column_name):
    """Extracts a cleaned name from the column name."""
    column_name = column_name.replace(" ", "").replace("-", "_")
    return column_name

def select_folder_path(prompt):
    # Open dialog to select the folder of interest and return its path
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    # Open a dialog to select a folder 
    folder_path = QFileDialog.getExistingDirectory(None, prompt, r"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset\Dataset_recategorized")
    return folder_path

# Define the directory path
input_folder = select_folder_path("Select the folder containing the Subject-RSFC maps, folder with SBA_GLM results")


# Automatically create an output folder in the same directory as the input folder
output_folder = os.path.join(os.path.dirname(input_folder), f"{os.path.basename(input_folder)}_Grouped_Betas")

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get a list of all Excel files in the directory
excel_files = [f for f in os.listdir(input_folder) if f.endswith(".xlsx") or f.endswith(".xls")]

data_df = pd.DataFrame()  # Initialize an empty DataFrame to store channel names

# Loop through each Excel file
for file in excel_files:
    file_path = os.path.join(input_folder, file)

    # load the data from the excel file that correspond to our input folder.
    if "Oxy" in input_folder:
        df = pd.read_excel(file_path, sheet_name="Oxy Results")
    elif "Deoxy" in input_folder:
        df = pd.read_excel(file_path, sheet_name="DeOxy Results")



    if file == excel_files[0]:
        # Extract the 2nd column to get the channel pair
        channels = df.iloc[1:,[1]].T.reset_index(drop=True).T.reset_index(drop=True)  # Reset index to avoid issues with concatenation
        channels.columns = ["Channels"]
        # Append the extracted column
        data_df = pd.concat([data_df, channels], axis=1)
        seed = df.columns[2].replace(" - ","_") # extract the seed name from 1st row 3rd column
        if "Oxy" in input_folder:
            seed = seed + "_Oxy"
        elif "Deoxy" in input_folder:
            seed = seed + "_DeOxy"



    # now append the betas of each excel file to the data_df.
    # the beta for oxy/deoxy is in 3rd column
    betas = df.iloc[1:, [2]].T.reset_index(drop=True).T.reset_index(drop=True)  # Reset index to avoid issues with concatenation

    # extract the participant number and place it as a column header
    participant_id = file.split("_")
    betas.columns = [participant_id[0]]
    data_df = pd.concat([data_df, betas], axis=1)

# Save the combined DF to an excel file. The output will be: Oxy/Deoxy seed x Participants
# in essence we have performed the GLM with the seed (Oxy/Deoxy) as a predictor. And then the individual results are grouped into a large dataframe, ready for group analysis (i.e., t-test).
output_path = os.path.join(output_folder, f"Group_GLM_Betas_{seed}.xlsx")
data_df.to_excel(output_path, index=False)

