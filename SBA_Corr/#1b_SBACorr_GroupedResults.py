"""
DESCRIPTION:
This script groups seed-based correlation analysis results from multiple 
subjects into consolidated Excel files for group-level analysis.

INPUT:
- Folder containing individual Excel files per subject with cross-correlation data
- Each Excel file should contain "Oxy Data" and "Deoxy Data" sheets
- Files should follow naming convention: "P##_*.xlsx" (e.g., "P01_CrossCorrelation.xlsx")

OUTPUT:
- Two separate Excel files containing grouped results:
  * SBA_Corr_Oxy_GroupedResults.xlsx: HbO correlation data for all subjects
  * SBA_Corr_Deoxy_GroupedResults.xlsx: HbR correlation data for all subjects
- Each output file contains channels as rows and participants as columns

PROCEDURE:
1. Iterate through all Excel files in the input directory
2. Extract participant ID from filename (assumes "P##_" format)
3. Load correlation matrices from "Oxy Data" and "Deoxy Data" sheets
4. Extract seed channel correlations:
   - HbO seed: S10-D7 (based on motor cortex GLM results)
   - HbR seed: S25-D23 (based on motor cortex GLM results)
5. Aggregate all participants' seed correlations into group matrices
6. Save consolidated results to separate Excel files

NOTES:
- Seed channels are hardcoded based on previous GLM analysis results
- First file processed determines the channel list structure
- Column names are cleaned (whitespace stripped) before processing
- Output folders are created automatically if they don't exist
"""


import os
import pandas as pd
from PySide6.QtWidgets import QApplication, QFileDialog
import sys
def select_folder_path(prompt):
    # Open dialog to select the folder of interest and return its path
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    # Open a dialog to select a folder 
    folder_path = QFileDialog.getExistingDirectory(None, prompt, r"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset")
    return folder_path


input_folder = select_folder_path("Select the folder containing the cross correlation results...")
# Example input folder path:
# r"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset\6_SBA_Corr\CrossCorrPerSubject"
output_folder_oxy = os.path.join(os.path.dirname(input_folder), "SBA_Corr_Oxy_GroupedResults")
output_folder_deoxy = os.path.join(os.path.dirname(input_folder), "SBA_Corr_Deoxy_GroupedResults")
if not os.path.exists(output_folder_oxy):
    os.makedirs(output_folder_oxy)
if not os.path.exists(output_folder_deoxy):
    os.makedirs(output_folder_deoxy)

grouped_oxy_data = pd.DataFrame()
grouped_deoxy_data = pd.DataFrame()

for file in os.listdir(input_folder):
    # Extract participant ID from the filename; to use for naming the output
    participant_id = file.split("_")[0] # Assuming the file name is like 'P01_CrossCorrelation.xlsx'
    # print(participant_id)

    # Read the oxy/deoxy data from the excel file
    oxy_data = pd.read_excel(os.path.join(input_folder, file), sheet_name="Oxy Data")
    deoxy_data = pd.read_excel(os.path.join(input_folder, file), sheet_name="Deoxy Data")
    # if its the first file, extract the channel names and place them in our group dataframes
    if grouped_oxy_data.empty and grouped_deoxy_data.empty:
        grouped_oxy_data = pd.concat([grouped_oxy_data, oxy_data.iloc[:, 0]], axis=1)
        grouped_deoxy_data = pd.concat([grouped_deoxy_data, deoxy_data.iloc[:, 0]], axis=1)
        grouped_oxy_data.columns = ['Channel']  # Rename the first column to 'Channel
        grouped_deoxy_data.columns = ['Channel']  # Rename the first column to 'Channel

    
    
    # Extract the seed correlation for HbO and HbR
    # find the column that contains the seed channel
    oxy_data.columns = oxy_data.columns.str.strip()
    deoxy_data.columns = deoxy_data.columns.str.strip()

    #! Hardcoded seed channels based on motor cortex GLM results
    # oxy seed: s10-d7
    # deoxy seed: s25-d23
    oxy_seed_correlation = oxy_data[['S10-D7']]  # Double brackets = DataFrame
    oxy_seed_correlation.columns = [participant_id]  # Rename the series to use participant ID as column name
    deoxy_seed_correlation = deoxy_data[['S25-D23']]
    deoxy_seed_correlation.columns = [participant_id]  # Use .name for Series
    # print(oxy_seed_correlation.head())
    grouped_oxy_data = pd.concat([grouped_oxy_data, oxy_seed_correlation], axis=1)
    # print(grouped_oxy_data)
    grouped_deoxy_data = pd.concat([grouped_deoxy_data, deoxy_seed_correlation], axis=1)



print(grouped_oxy_data.head())
print(grouped_deoxy_data.head())
# Save the grouped results to an excel file in their respective folders
output_path_oxy = os.path.join(output_folder_oxy, "SBA_Corr_Oxy_GroupedResults_S10D7.xlsx")
output_path_deoxy = os.path.join(output_folder_deoxy, "SBA_Corr_Deoxy_GroupedResults_S25D23.xlsx")
grouped_oxy_data.to_excel(output_path_oxy, index=False)
grouped_deoxy_data.to_excel(output_path_deoxy, index=False)
print(f"Grouped results saved to {output_path_oxy} and {output_path_deoxy}")
