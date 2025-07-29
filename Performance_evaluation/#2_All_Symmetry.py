"""
===============================================================================
HbO-HbR SIMILARITY OF T-MAPS ACROSS METHODS
===============================================================================

Purpose: Evaluate symmetry between oxygenated (HbO) and deoxygenated (HbR) 
hemoglobin RSFC t-maps by correlating their t-values across different methods.

Perfect symmetry appears as points close to the diagonal line from (0,0) 
to (1,1) after normalization. Closer points = more symmetric performance.

Methods Analyzed:
    • SBA-GLM, SBA-GLM-Resp, SBA-Corr (separate HbO/HbR files)
    • ICA-Logcosh, ICA-Skew (single file with separate sheets)

Processing:
1. Interactive file selection with validation
2. Remove short channels (detector > 28) and seed channels for SBA
3. Channel alignment verification between HbO/HbR
4. For ICA: Sign alignment check with motor task GLM + HbR sign flip
5. Pearson correlation calculation
6. MinMax normalization [0,1] for visualization
7. Square scatter plots with diagonal symmetry line

Output:
    • {Method}_HbR_HbO_Similarity.png: Correlation scatter plot
    • {Method}_HbR_HbO_Similarity.txt: Statistical results
    • Saved to: 8_PerformanceEvaluation/Similarity/

Interpretation:
    • Points near diagonal: Good HbO-HbR RSFC t-map similarity
    • Correlation coefficient quantifies similarity strength
===============================================================================

For convinience the excel files with the t-values are in a single folder
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
from PySide6.QtWidgets import QApplication, QFileDialog
import sys

def check_sign_alignment_with_motor_task(oxy_data, deoxy_data):
    """
    Check if the oxy and deoxy data are aligned with the motor task glm activation map, if they are then invert the sign of the deoxy data (the deoxy activation map is negative)
    Args:
        oxy_data (pd.DataFrame): DataFrame containing the oxy t-values.
        deoxy_data (pd.DataFrame): DataFrame containing the deoxy t-values.
    Returns:
        bool: True if oxy and deoxy are aligned with motor task glm activation map, False otherwise.
    """
    # input of the motor task glm activation map




    # Extract the motor task t values for HbO and HbR from the group GLM results
    # oxy & deoxy
    motor_file_path = r"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset\3_Data_PrePro\02a_test_MotorAction_GroupGLM_correctSerialC\MultiStudy_GLM_Results_MSGLM_Results_CorrectSC.xlsx"
    motor_oxy_df = pd.read_excel(motor_file_path ,sheet_name='Oxy Contrast Results')
    motor_deoxy_df = pd.read_excel(motor_file_path, sheet_name='Deoxy Contrast Results')
    motor_oxy_df=clean_motor_df(motor_oxy_df, heme_type='hbo')
    motor_deoxy_df=clean_motor_df(motor_deoxy_df, heme_type='hbr')

    # check if the channels match
    if not check_if_channels_match(motor_oxy_df, motor_deoxy_df):
        print("Channels do not match between motor oxy and deoxy dataframes. Please check the files or your code")
        quit()
    else:
        print("Channels match between motor oxy and deoxy dataframes. Proceeding with sign alignment check.")

    oxy_pearson_r, _ = stats.pearsonr(motor_oxy_df['T-Value'], oxy_data['T-Value'])
    deoxy_pearson_r, _ = stats.pearsonr(motor_deoxy_df['T-Value'], deoxy_data['T-Value'])
    if oxy_pearson_r < 0 or deoxy_pearson_r < 0: # if the correlation is negative then we have a problem and we will quit the script to revise our thought process; but sign alignment has already be done in previous scripts so it should never be not aligned
        print("The oxy or deoxy data are not aligned with the motor task glm activation map. Please check the files or your code")
        quit()
    else:
        print("The oxy and deoxy data are aligned with the motor task glm activation map. Proceeding with sign alignment check.")
        return True
    # As expected the oxy AND deoxy data are aligned with the motor task glm activation map


def clean_motor_df(df, heme_type=str):
    """Clean the motor task t-values DataFrame based on the heme type (HbO or HbR).
    Args:
        df (pd.DataFrame): The DataFrame containing the motor task t-values.
        heme_type (str): The type of heme ('hbo' for HbO, 'hbr' for HbR).
    """
    df = df.iloc[1:, [1, 2]]  # channel names in column B; t values in column C
    df.columns = ["Channel", "T-Value"]


    df = df[df["Channel"].apply(lambda x: int(extract_d_value(x)) <= 28)]  # Remove short channels
    df["T-Value"] = df["T-Value"].astype(float)  # make them float64
    return df.reset_index(drop=True)  


def select_folder_path(prompt):
    
# Copy/pasted from Michael Luhrs


    # Open dialog to select the folder of interest and return its path
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    # Open a dialog to select a folder 
    folder_path = QFileDialog.getExistingDirectory(None, prompt, r"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset")
    return folder_path

def select_file_path(prompt):
    """Open dialog to select the file of interest and return its path."""
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    # Open a dialog to select a file
    file_path, _ = QFileDialog.getOpenFileName(None, prompt, r"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset\8_PerformanceEvaluation")
    return file_path

def check_if_channels_match(oxy_df, deoxy_df):
    """Check if the channels in both dataframes match."""
    return set(oxy_df['Channel']) == set(deoxy_df['Channel'])
    
def extract_d_value(channel):
    """Extract the d value from the channel name.
    Example:
        Input: 'S10-D7'
        Output: '7'
    """
    channel = channel.split('-')[-1]  # Extract the last part after the underscore, which is the: D##
    return channel[1:] # Extract the last part after the underscore, which is the d value

def remove_short_channel_names(df):
    """Remove short channel names from the dataframe and return it"""
    df = df[df['Channel'].apply(lambda x: int(extract_d_value(x)) <= 28)]
    # reset the index of the dataframe
    df = df.reset_index(drop=True)
    return df

def plot_linear_correlation(oxy_data, deoxy_data, oxy_data_name, deoxy_data_name, method, pearson_r, p_value, file_path):
    """Plot the linear correlation between oxy and deoxy t-values.
    Args:
        oxy_data (pd.DataFrame): DataFrame containing the oxy t-values.
        deoxy_data (pd.DataFrame): DataFrame containing the deoxy t-values.
        oxy_data_name (str): Name of the oxy data file.
        deoxy_data_name (str): Name of the deoxy data file.
        method (str): Method name for the plot title.
    Normalize the t-values to the range [0, 1] for better visualization and to be able to plot the symmetry based on a diagonal line
        closer to diagonal line means more symmetry          
    """
    # Normalize the t-values to the range [0, 1]

    # min max normalization with scikit-learn
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    # for debuging purposes; verification of the minmax normalization it works correctly
    # # find the min and max of the t-values
    # oxy_data_min = oxy_data["T-Value"].min()
    # oxy_data_max = oxy_data["T-Value"].max()
    # # the channel of the min and max t-values
    # oxy_data_min_channel = oxy_data.loc[oxy_data["T-Value"].idxmin(), "Channel"]
    # oxy_data_max_channel = oxy_data.loc[oxy_data["T-Value"].idxmax(), "Channel"]
    # print(f"Oxy data min: {oxy_data_min} at channel {oxy_data_min_channel}")
    # print(f"Oxy data max: {oxy_data_max} at channel {oxy_data_max_channel}")

    oxy_data["T-Value"] = scaler.fit_transform(oxy_data[["T-Value"]])
    deoxy_data["T-Value"] = scaler.fit_transform(deoxy_data[["T-Value"]])

    # lets plot now the t-values of the oxy against the t values of the deoxy
    plt.figure(figsize=(8, 8)) 
    plt.scatter(oxy_data["T-Value"], deoxy_data["T-Value"], alpha=0.5, color='black', label=f"{oxy_data_name} vs {deoxy_data_name}\n r = {pearson_r:.2f}, p = {p_value:.2e}")
    # Add the perfect symmetry line (diagonal)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='Perfect Symmetry Line')
    # Set equal aspect ratio to ensure square plotting area
    plt.gca().set_aspect('equal', adjustable='box')
    # Set axis limits to ensure square bounds
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel(f"Oxy T-Values (normalized)", fontsize=14)
    plt.ylabel(f"Deoxy T-Values (normalized)", fontsize=14)
    plt.title(f'{method}: HbO vs HbR Similarity Analysis', fontsize=16, fontweight='bold')
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, alpha=0.3)  
        
    # Use tight_layout to remove extra whitespace
    plt.tight_layout()     
    # Save the plot to a file
    plot_file_path = os.path.join(output_file_path, f"{method}_HbR_HbO_Similarity.png")
    plt.savefig(plot_file_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Save the text file with the results
    # write results in text file
    results_txt = []
    results_txt.append(f"Method: {method}\n")
    results_txt.append(f"Oxy File: {oxy_data_name}\n")
    results_txt.append(f"Deoxy File: {deoxy_data_name}\n")
    results_txt.append(f"Pearson correlation coefficient: {pearson_r:.2f}\n")
    results_txt.append(f"P-value: {p_value:.2e}\n")
    results_txt_name = f"{method}_HbR_HbO_Similarity.txt"
    with open(os.path.join(output_file_path, results_txt_name), 'w') as f:
        f.writelines(results_txt)
    print(f"Results saved to {os.path.join(output_file_path, results_txt_name)}")

# Instead of itterating through the data folder we are going to select the files of interest
# SBA files are processed differently than ICA files
# in SBA the excel for oxy and deoxy are in separate files
# in ICA the excel for oxy and deoxy are in the same file under a different sheet name
output_file_path = r"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset\8_PerformanceEvaluation\Similarity"

methods = [
    "SBA-GLM",
    "SBA-GLM-Resp",
    "SBA-Corr",
    "ICA-Logcosh",
    "ICA-Skew"
]

for method in methods:
    print(f"Processing method: {method}")
    # if method != "ICA-Logcosh":
    #     # skipping for debugging purposes
    #     continue

    # Select the excel files of the SBA-GLM, SBA-GLM-Resp, SBA-Corr
    if "SBA" in method:
        while True:
            # Select the oxy file
            oxy_file = select_file_path(f"Select the oxy file for {method}")
            oxy_file_name = os.path.basename(oxy_file)
            oxy_file_name = oxy_file_name.replace(".xlsx", "").replace("_Group", "").replace("_", "-")
            
            # Select the deoxy file
            deoxy_file = select_file_path(f"Select the deoxy file for {method}")
            deoxy_file_name = os.path.basename(deoxy_file)
            deoxy_file_name = deoxy_file_name.replace(".xlsx", "").replace("_Group", "").replace("_", "-")
            if method not in oxy_file_name or method not in deoxy_file_name or "Oxy" not in oxy_file_name or "Deoxy" not in deoxy_file_name:
                print(f"Please select the correct files for {method} for {oxy_file_name} and {deoxy_file_name}. The files should contain the method name in their name.")
                continue
            else:
                break


        oxy_data = pd.read_excel(oxy_file)
        deoxy_data = pd.read_excel(deoxy_file)
        print(oxy_data.head())
        oxy_data = remove_short_channel_names(oxy_data)
        deoxy_data = remove_short_channel_names(deoxy_data)
        # Keep only columns with channels and t-values
        oxy_data = oxy_data[['Channel', 'T-Value']]
        deoxy_data = deoxy_data[['Channel', 'T-Value']]
        # print(oxy_data.head())


        # Here we are going to drop the seed channels for SBA analysis. (i) t-values of the seed channels are not reliable they are derived from self comparison, and thus do not reflect on RSFC. (ii) due to their nature they are near infinite and this will skew the results, and on some cases the correlation doesnt work because the number is in string format: 'inf'
        # when dropping the seed make sure to drop it from both oxy/deoxy,
        # Remove seed: S10-D7 and S25-D23 from both oxy and deoxy dataframes
        oxy_data = oxy_data[~oxy_data['Channel'].isin(['S10-D7', 'S25-D23'])]
        deoxy_data = deoxy_data[~deoxy_data['Channel'].isin(['S10-D7', 'S25-D23'])]
        
        # Reset index
        oxy_data = oxy_data.reset_index(drop=True)
        deoxy_data = deoxy_data.reset_index(drop=True)

        # Correlate the t_values of oxy and deoxy after ensuring that the channel order is the same in both of them
        if not check_if_channels_match(oxy_data, deoxy_data):
            print("Channels do not match between oxy and deoxy dataframes. Please check the files or your code")
            quit()
        else:
            print("Channels match between oxy and deoxy dataframes. Proceeding with correlation.")

        pearson_r, p_value = stats.pearsonr(oxy_data['T-Value'], deoxy_data['T-Value']) # for the pearsonr to work we need to pass the values as numpy arrays
        print(f"Pearson correlation coefficient: {pearson_r}, p-value: {p_value}")





        plot_linear_correlation(oxy_data=oxy_data, 
                                deoxy_data=deoxy_data, 
                                oxy_data_name=oxy_file_name, 
                                deoxy_data_name=deoxy_file_name, 
                                method=method, 
                                pearson_r= pearson_r, 
                                p_value=p_value,
                                file_path=output_file_path)


    else:
        while True:

            # Select the excel file for the ICA method
            ica_file = select_file_path(f"Select the SINGLE file for {method}")
            ica_file_name = os.path.basename(ica_file)
            ica_file_name = ica_file_name.replace(".xlsx", "").replace("_Group", "").replace("_", "-")
            if method not in ica_file_name:
                print(f"Please select the correct file for {method}. The file should contain the method name in its name.")
                continue
            else:
                break


        data_oxy = pd.read_excel(ica_file, sheet_name='Oxy Group RSFC')
        data_deoxy = pd.read_excel(ica_file, sheet_name='Deoxy Group RSFC')
        data_oxy["Channel"] = data_oxy["Channel"].str.replace('hbo', '').str.replace('_','-').str.strip()  # Clean the channel names to match the other dataframes
        data_deoxy["Channel"] = data_deoxy["Channel"].str.replace('hbr', '').str.replace('_','-').str.strip()

        # Short channels are not in the ICA data so we dont need to remove them
        # Keep only columns with channels and t-values
        data_oxy = data_oxy[['Channel', 'T-Value']]
        data_deoxy = data_deoxy[['Channel', 'T-Value']]
        print(data_oxy.head())
        # no need to drop any channels for ICA
        # Correlate the t_values of oxy and deoxy after ensuring that the channel order is the same in both of them
        if not check_if_channels_match(data_oxy, data_deoxy):
            print("Channels do not match between oxy and deoxy dataframes. Please check the files or your code")
            quit()
        else:
            print("Channels match between oxy and deoxy dataframes. Proceeding with correlation.")

        # Now we need to place the oxy and deoxy data in the same direction
        # to do so we are going to see if they are aligned with the motor task glm activation map; which they are
        # then we need to flip the deoxy data, therefore, matching the oxy data directionality
        # we do this because when we do the minmax normalization the values of -10 (maximal activation) will be 0 (minimal) and the values of +10 (maximal) will be 1 (maximal). Min Max normalization introduces this problem
        # in terms of correlation the p value doesnt matter a lot if its negative or positive.
        
        # check alignment with motor task glm activation map




        if not check_sign_alignment_with_motor_task(data_oxy, data_deoxy): # if not true the script has already eneded from the helper function
            quit()
        
        # Now that we know that the oxy and deoxy data are aligned with the motor task glm activation map we can proceed with our thinking process
        # We have to flip the deoxy data to match the oxy data directionality
        data_deoxy['T-Value'] = -data_deoxy['T-Value']

        pearson_r, p_value = stats.pearsonr(data_oxy['T-Value'], data_deoxy['T-Value']) # for the pearsonr to work we need to pass the values as numpy arrays      
        print(f"Pearson correlation coefficient: {pearson_r}, p-value: {p_value}")

        plot_linear_correlation(oxy_data=data_oxy,
                                deoxy_data=data_deoxy, 
                                oxy_data_name=ica_file_name + "-Oxy", 
                                deoxy_data_name=ica_file_name + "-Deoxy", 
                                method=method, 
                                pearson_r= pearson_r, 
                                p_value=p_value,
                                file_path=output_file_path)
    # Select the excel files of the ICA-Logcosh, ICA-Skew