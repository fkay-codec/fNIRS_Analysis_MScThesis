"""
Cross-Correlation Analysis for fNIRS Data

This script performs cross-correlation analysis on functional Near-Infrared Spectroscopy (fNIRS) data.
It processes .snirf files that contain preprocessed fNIRS data in cross-correlation (CC) format.

What this script does:
1. Allows user to select a folder containing multiple .snirf files
2. For each .snirf file, it computes correlation matrices for:
   - Oxyhemoglobin (HbO) channels
   - Deoxyhemoglobin (HbR) channels
3. Saves the results as Excel files with formatted correlation matrices
4. Creates a separate output folder for all results

Input Requirements:
- .snirf files must be in CC format

Output:
- Excel files with two sheets: 'Oxy Data' and 'Deoxy Data'
- Each sheet contains a formatted correlation matrix
- Channel names are cleaned (underscores replaced with hyphens)
- Output files are named: {original_filename}_CrossCorrelation.xlsx

Usage:
1. Run the script
2. Select the folder containing your .snirf files
3. Select the output folder where results will be saved
4. The script will process all files and create results in a new folder

Date: Created May 25, 2025
"""


import mne
import pandas as pd
import os
import sys
import glob
from PySide6.QtWidgets import QApplication, QFileDialog

def select_folder_path(prompt):
    # Open dialog to select the folder of interest and return its path
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    # Open a dialog to select a folder 
    folder_path = QFileDialog.getExistingDirectory(None, prompt, r"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset")
    return folder_path

def correlation_analysis_per_subject(input_file, output_file):
    """
    Performs correlation analysis on a single subject's fNIRS data.
    
    This function processes one .snirf file containing fNIRS data and computes correlation 
    matrices for both oxyhemoglobin (HbO) and deoxyhemoglobin (HbR) channels. The results 
    are formatted and saved to an Excel file with two separate sheets.
    
    Processing steps:
    1. Loads the .snirf file using MNE-Python
    2. Extracts HbO and HbR channel data separately
    3. Computes Pearson correlation matrices for each hemoglobin type
    4. Formats the correlation matrices for better readability:
       - Replaces underscores with hyphens in channel names
       - Removes 'hbo'/'hbr' suffixes from channel names
       - Adds descriptive titles in the first cell
    5. Saves both matrices to Excel with separate sheets
    
    Parameters:
    -----------
    input_file : str
        Full path to the input .snirf file containing preprocessed fNIRS data
        Must be in cross-correlation (CC) format with HbO and HbR channels
        
    output_file : str
        Full path for the output Excel file where correlation matrices will be saved
        Will contain two sheets: 'Oxy Data' (HbO) and 'Deoxy Data' (HbR)

    """
    raw = mne.io.read_raw_snirf(input_file, preload=True)
    # --- HbO ---
    # correlate the hbo channels
    hbo_picks = mne.pick_types(raw.info, fnirs='hbo')
    hbo_channel_names = [raw.ch_names[i] for i in hbo_picks]
    hbo_timeseries= raw.get_data(picks=hbo_picks)  # shape: (n_channels, n_samples)
    # Build a DataFrame: Columns are channels, rows are time points
    hbo_df = pd.DataFrame(hbo_timeseries.T, columns = hbo_channel_names)
    # print("hbo_df:")
    # print(hbo_df.head())
    oxy_correlation_df = hbo_df.corr()
    # print(oxy_correlation_df.head())
    # quit()


    # Clean up the correlation DataFrame from the header and index, and clean up the channel names + add a title in first cell
    # with this we drop the index and reset the column names, but we keep the information as non index
    oxy_correlation_df = oxy_correlation_df.reset_index()
    oxy_correlation_df = oxy_correlation_df.T.reset_index()

    oxy_correlation_df.iloc[0, 1:] = oxy_correlation_df.iloc[0, 1:].str.replace('_', '-').str.replace('hbo', '')
    oxy_correlation_df.iloc[1:, 0] = oxy_correlation_df.iloc[1:, 0].str.replace('_', '-').str.replace('hbo', '')
    oxy_correlation_df.iloc[0,0] = "Correlations for oxy-Hb"
    # Reset the column index and drop it to make it cleaner
    oxy_correlation_df = oxy_correlation_df.T.reset_index(drop=True).T
    # print(oxy_correlation_df.head())
    # quit()

    # --- HbR ---
    hbr_picks = mne.pick_types(raw.info, fnirs='hbr')
    hbr_channel_names = [raw.ch_names[i] for i in hbr_picks]
    hbr_timeseries = raw.get_data(picks=hbr_picks)  # shape: (n_channels, n_samples)
    hbr_df = pd.DataFrame(hbr_timeseries.T, columns = hbr_channel_names)
    # print("hbr_df:")
    # print(hbr_df.head())
    hbr_correlation_df = hbr_df.corr()

    # print("Correlation DataFrame for hbr:")
    # print(hbr_correlation_df.head())
    # Clean up the HbR correlation DataFrame from the header and index, and clean up the channel names + add a title in first cell
    hbr_correlation_df = hbr_correlation_df.reset_index()
    hbr_correlation_df = hbr_correlation_df.T.reset_index()
    hbr_correlation_df.iloc[0, 1:] = hbr_correlation_df.iloc[0, 1:].str.replace('_', '-').str.replace('hbr', '')
    hbr_correlation_df.iloc[1:, 0] = hbr_correlation_df.iloc[1:, 0].str.replace('_', '-').str.replace('hbr', '')
    hbr_correlation_df.iloc[0,0] = "Correlations for deoxy-Hb"
    hbr_correlation_df = hbr_correlation_df.T.reset_index(drop=True).T
    # print("Cleaned HbR correlation DataFrame:")
    # print(hbr_correlation_df.head())

    with pd.ExcelWriter(output_file) as writer:
        oxy_correlation_df.to_excel(writer, sheet_name='Oxy Data',index=False, header=False)
        hbr_correlation_df.to_excel(writer, sheet_name='Deoxy Data',index=False, header=False)
    print(f"Correlation matrices saved to {output_file}")

def process_folder(input_folder):
    """
    Batch processes all .snirf files in a specified folder for correlation analysis.
    
    This function prepares the correlation analysis workflow for multiple subjects by:
    1. Scanning the input folder for all .snirf files
    2. Processing each file individually using correlation_analysis_per_subject()
    3. Saving the results in a new selected output folder

    The function automatically creates a results folder alongside the input folder with
    a descriptive name that includes "_CorrelationResultsPerSubject" suffix.
    
    Parameters:
    -----------
    input_folder : str
        Full path to the directory containing .snirf files to be processed
        All .snirf files in this folder will be automatically detected and processed
 
    Output Structure:
    -----------------
    - Output folder: {input_folder_name}_CorrelationResultsPerSubject/
    - Output files: {original_filename}_CrossCorrelation.xlsx
    """
    
    # Select the output folder where the results will be saved
    output_folder = select_folder_path("Select the folder you want the results to be saved in...")
    # create a subfolder in the output folder, named as cross wise correlation
    output_folder = os.path.join(output_folder, "CrossCorrPerSubject")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Find all .snirf files in the input folder
    snirf_files = glob.glob(os.path.join(input_folder, "*.snirf"))
    
    if not snirf_files:
        print("No .snirf files found in the input folder.")
        return
    
    print(f"Found {len(snirf_files)} .snirf files to process.")

    # Process each .snirf file
    for snirf_file in snirf_files:
        print(f"\nProcessing file: {os.path.basename(snirf_file)}")
        
        # Create output filename: input_filename + "_CrossCorrelation.xlsx"
        base_filename = os.path.splitext(os.path.basename(snirf_file))[0].split('_')[0]
        # print(base_filename)
        # quit()
        output_filename = f"{base_filename}_CrossCorrelation.xlsx"
        output_file = os.path.join(output_folder, output_filename)
        
        try:
            correlation_analysis_per_subject(snirf_file, output_file)
            print(f"Successfully processed: {os.path.basename(snirf_file)}")
        except Exception as e:
            print(f"Error processing {os.path.basename(snirf_file)}: {str(e)}")



input_folder = select_folder_path("Select the folder containing the .snirf files in CC format")
if not input_folder:
    print("No folder selected. Exiting.")
    sys.exit()

process_folder(input_folder)
print("\nProcessing complete!")