import pandas as pd
import os
from PySide6.QtWidgets import QApplication, QFileDialog
import sys
import scipy.stats as stats
from statsmodels.stats.multitest import fdrcorrection


def select_folder_path(prompt):
    # Open dialog to select the folder of interest and return its path
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    # Open a dialog to select a folder 
    folder_path = QFileDialog.getExistingDirectory(None, prompt, r"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset")
    return folder_path

input_folder = select_folder_path("Select the folder containing the grouped...")
# Example input folder path:
# r"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset\6_SBA_Corr\SBA_Corr_Oxy_GroupedResults"
# r"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset\6_SBA_Corr\SBA_Corr_Deoxy_GroupedResults"


# Iterate through each file in the input folder
## its only one file in the input folder currently... so its a bit unnecessary to loop through it.
### this code here is copy/pasted from previous ttest from folder sba-ttest; script #2 SBA ttest.py
for file_name in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file_name)
  
    # Skip non-Excel files
    if not file_name.endswith(".xlsx"):
        print(f"Skipping non-Excel file: {file_name}")
        continue
    # skip files that are already processed with t-test
    if file_name.endswith("_ttest.xlsx"):
        print(f"Skipping already processed file: {file_name}")
        continue

    # Correctly construct the new file name with the .xlsx extension
    new_file_name = f"{os.path.splitext(file_name)[0]}_ttest.xlsx"
    output_file = os.path.join(input_folder, new_file_name)

    # Read the Excel file, in the format: channel_name, P01_betas, ..., PXX_rvalue
    data = pd.read_excel(file_path)
    print(data.head())
  

    # Initialize lists to store results
    channel_names = []
    t_values = []
    p_values = []
    adjusted_p_values = []
    std_devs = []
    mean_r_values = []
    cohen_ds = []

    # Iterate through each row in the DataFrame
    for index, row in data.iterrows():
        # Extract the channel name (first column)
        channel_name = row.iloc[0].strip()
        # check if the channel name has any whitespace and strip it

        # Extract the r values (skip the first column)
        r_values = pd.to_numeric(row[1:], errors='coerce')  # Convert to numeric, coercing errors to NaN

        # Check for NaN values and skip the row if necessary
        if r_values.isna().any():
            print(f"Warning: NaN values detected in channel {channel_name}. Skipping this row.")
            continue
        
        # Calculate the standard deviation of beta values and mean
        std_dev = stats.tstd(r_values)
        mean_r = r_values.mean() # double checked if its correct

        # Perform one-sample t-test (null hypothesis: r = 0)
        t_stat, p_val = stats.ttest_1samp(r_values, 0)

        # Compute Cohens'd
        cohen_d = (mean_r - 0) / std_dev # (mean beta - theoretical mean (0) ) / std deviation
        
        # Store the results
        channel_names.append(channel_name)
        t_values.append(t_stat)
        p_values.append(p_val)
        std_devs.append(std_dev)
        mean_r_values.append(mean_r)     
        cohen_ds.append(cohen_d)  

    # Apply FDR adjustment to the collected p-values
    #! Not sure if its needed, should check
    adjusted_p_values = stats.false_discovery_control(p_values, method='bh')  # Use Benjamini-Hochberg method

    # Create a DataFrame with the results
    results_df = pd.DataFrame({
        "Channel": channel_names,
        "T-Value": t_values,
        "Raw P-Value": p_values,
        "FDR Adjusted P-Value": adjusted_p_values,
        "Standard Deviation": std_devs,
        "Mean Beta": mean_r_values,
        "Cohen's d": cohen_ds      
    })

    # Export the DataFrame to an Excel file
    results_df.to_excel(output_file, index=False, header=True)
    print(f"Results exported to {output_file}")
