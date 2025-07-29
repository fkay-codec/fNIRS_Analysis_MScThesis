"""
Description:
This script evaluates the performance of different methods (e.g., SBA, ICA) using Receiver Operating Characteristic (ROC) analysis. 
It processes multiple datasets, normalizes the data, and generates two sets of ROC curves:
1. ROC curves using the **motor activation map** as the golden standard.
2. ROC curves using the **fOLD map** as the golden standard.

Distinct colors are dynamically assigned to each curve for better visualization,
        
Some Theory on ROC
    ROC is a probability curve that plots sensitivity and specificity. 
        True positive rate  (correctly predicted positive) vs.
        False positive rate (incorrectly predicted positive)

    Important terms specificity and sensitivity
    Specificity: True Negative (TN) - the proportion of actual negatives that are correctly identified.
    Sensitivity: True Positive Rate (TPR) - proportion of actual positives that are correctly identified.
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

ROC curve is a graphical representation of the trade-off between sensitivity and specificity across different thresholds.
In our case the golden standard is the motor task activation map & the fOLD predefined map [channels to Brodmann Areas groupped together to form broad anatomical areas]
We want to see how well our RSFC values can predict this activation.
    Binarize the motor task activation map:
        - if significant activation in channel: 1
        - if not significant activation :       0
    
    Binarize the fOLD predefined map:
        - if channel is in the motor areas: 1
        - if channel is not in the motor areas: 0

Input of specific script:
    - Motor task activation map (binarized)
    - fOLD predefined map (binarized)
    - RSFC values from SBA (oxy/deoxy)
        - Seed: S10D7 for oxy, S25D24 for deoxy
        - ttest excel
        - RFX excel
    - RSFC values from ICA (oxy/deoxy)
        - based on different tolerance levels in the FastICA algorithm

Steps in the Code:
1. Load and preprocess the datasets.
2. Normalize the 'Effsize' column in each dataframe.
3. Dynamically generate distinct colors for each analysis using a colormap.
4. Compute two sets of ROC curves:
   a. One set using the motor activation map as the golden standard.
   b. Another set using the fOLD map as the golden standard.
5. Plot the ROC curves with proper labels, legends, and distinct colors.
6. Add titles, subtitles, and axis labels for better visualization.

NOTE: motor task activation map is FDR corrected, so the p-values are corrected for multiple comparisons.


NOTE: This script was designed initially to create ROC curves for fOLD and motor task with no consideration of a stricter p threshold; so in its initial form the motor task map was binirized on p<=0.05; to create the figure with the motor map thresholded at p<=00.1 i just run the script while changing the p-value threshold and the ROC title. 
"""
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import colormaps
import random
import numpy as np
import pandas as pd
import os
import scipy.stats as stats


def normalize_dataframes(dataframes):
    """Normalize the dataframes to have values between 0 and 1 using min-max normalization. By turning them to numpy and then return numpy arrays"""
    scaler = MinMaxScaler()
    for i, (name, df) in enumerate(dataframes):
        # print(name)
        # print(df.head())
        # Normalize the 'Effsize' or 'Beta' column
        df['Effsize'] = scaler.fit_transform(df[['Effsize']])
        # print(f"Normalized {name} dataframe.")
        # print(df.head())
        # print("-" * 50)
        # print(f"Min: {df['Effsize'].min()}, Max: {df['Effsize'].max()}")
        # quit()
    return dataframes

def correct_inf(df, values=str):
    """Check if there are any infinity values in the datafrae and correct them by replacing them with the maximum finite value in the dataframe."""
    values_array = np.array(df[values])
    max_val = np.max(values_array)
    # min_val = np.min(values_array)
    
    sorted_values = np.sort(values_array)
    second_max_val = sorted_values[-2]  # Get the second maximum value
    values_array[values_array == max_val] = second_max_val  # Replace the maximum value with the second maximum value
    df[values] = values_array  # Update the dataframe with the corrected values
    return df

def check_inf(df, values=str):
    """Check if there are any infinity values in the dataframe and return True if found, False otherwise."""
    values_array = np.array(df[values])
    if np.isinf(values_array).any():
        return True
    else:
        return False

# Create multiple ROC curves for motor task as golden standard for both oxy and deoxy analyses.
def plot_multiple_roc_curves(golden_oxy, golden_deoxy, oxy, deoxy, goldenstandard):
    """
    Plot multiple ROC curves for the given analyses.
    Input:
        golden: the golden standard dataframe
        oxy: list of tuples (name, dataframe) for oxy analyses
        deoxy: list of tuples (name, dataframe) for deoxy analyses
    Output:
        A plot with multiple ROC curves for the given analyses.
    """
    # Create the figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    # Oxy ROC curves
    y_true_oxy = golden_oxy['Effsize'].astype(float)

    # get a palate of colors for the curves using seaborn
    # Here we split the color palette into two equal parts to be used for oxy/deoxy
    colors = sns.color_palette("hls", len(oxy))  # Generate distinct colors for each analysis
    y_scores_oxy = []
    for (name, df), color in zip(oxy, colors):
        y_scores_oxy.append((name, df['Effsize'].astype(float), color))
    all_oxy_results = []
    best_five_oxy = []
    for name, score, color in y_scores_oxy:
        fpr, tpr, _ = roc_curve(y_true_oxy, score)  # Ensure the scores are float for ROC calculation
        roc_auc = auc(fpr, tpr)
        all_oxy_results.append((fpr, tpr, color, name, roc_auc))

    best_five_oxy = sorted(all_oxy_results, key=lambda x: x[4], reverse=True)  # Sort by AUC in descending order
    # Plot the best 5 curves
    for fpr, tpr, color, name, roc_auc in best_five_oxy[:5]:
        # print(f"Plotting {name} with AUC: {roc_auc:.3f}")
        ax1.plot(fpr, tpr, color=color, lw=2, alpha =0.8, label=f'{name} (AUC = {roc_auc:.3f})')

    # # Example output for debugging
    # for name, scores, color in y_scores_oxy:
    #     print(f"Name: {name}, Color: {color}")        

    ax1.set_title(f'ROC Curves: Oxy Analyses\n{goldenstandard}', loc='center')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='Chance level')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.legend(loc="lower right", fontsize='small')
    ax1.grid(True, alpha=0.3)

    # Deoxy ROC curves
    y_true_deoxy = golden_deoxy['Effsize'].astype(float)
    y_scores_deoxy = []
    best_five_deoxy = []
    all_deoxy_results = []
    for (name, df), color in zip(deoxy, colors):
        y_scores_deoxy.append((name, df['Effsize'].astype(float), color))
    for name, score, color in y_scores_deoxy:
        fpr, tpr, _ = roc_curve(y_true_deoxy, score)
        roc_auc = auc(fpr, tpr)
        all_deoxy_results.append((fpr, tpr, color, name, roc_auc))
    best_five_deoxy = sorted(all_deoxy_results, key=lambda x: x[4], reverse=True)  # Sort by AUC in descending order
    # Plot the best 5 curves
    for fpr, tpr, color, name, roc_auc in best_five_deoxy[:5]:
            ax2.plot(fpr, tpr, color=color, lw=2.5, alpha = 0.7, label=f'{name} (AUC = {roc_auc:.3f})')
    ax2.set_title(f'ROC Curves: Deoxy Analyses\n{goldenstandard}', loc = 'center')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='Chance level')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.legend(loc="lower right", fontsize='small')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    # Prepare results text for saving
    results_text = []
    results_text.append("=" * 80)
    results_text.append(f"ROC ANALYSIS RESULTS - {goldenstandard.upper()}")
    results_text.append("=" * 80)
    results_text.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results_text.append(f"Golden Standard: {goldenstandard}")
    results_text.append(f"Total Oxy Analyses: {len(all_oxy_results)}")
    results_text.append(f"Total Deoxy Analyses: {len(all_deoxy_results)}")
    results_text.append("")
    
    # Add detailed AUC results
    results_text.append("DETAILED AUC SCORES:")
    results_text.append("-" * 50)
    results_text.append("OXY ANALYSES (sorted by AUC):")
    results_text.append("-" * 30)
    sorted_oxy = sorted(all_oxy_results, key=lambda x: x[4], reverse=True)
    for i, (fpr, tpr, color, name, roc_auc) in enumerate(sorted_oxy, 1):
        results_text.append(f"{i:2d}. {name:<40} AUC = {roc_auc:.5f}")
    
    results_text.append("")
    results_text.append("DEOXY ANALYSES (sorted by AUC):")
    results_text.append("-" * 30)
    sorted_deoxy = sorted(all_deoxy_results, key=lambda x: x[4], reverse=True)
    for i, (fpr, tpr, color, name, roc_auc) in enumerate(sorted_deoxy, 1):
        results_text.append(f"{i:2d}. {name:<40} AUC = {roc_auc:.5f}")
    
    # Add statistical summary
    results_text.append("")
    results_text.append("STATISTICAL SUMMARY:")
    results_text.append("-" * 50)
    oxy_aucs = [x[4] for x in all_oxy_results]
    deoxy_aucs = [x[4] for x in all_deoxy_results]
    
    results_text.append(f"Oxy AUC Statistics:")
    results_text.append(f"  Mean: {np.mean(oxy_aucs):.5f}")
    results_text.append(f"  Std:  {np.std(oxy_aucs):.5f}")
    results_text.append(f"  Min:  {np.min(oxy_aucs):.5f}")
    results_text.append(f"  Max:  {np.max(oxy_aucs):.5f}")
    
    results_text.append(f"Deoxy AUC Statistics:")
    results_text.append(f"  Mean: {np.mean(deoxy_aucs):.5f}")
    results_text.append(f"  Std:  {np.std(deoxy_aucs):.5f}")
    results_text.append(f"  Min:  {np.min(deoxy_aucs):.5f}")
    results_text.append(f"  Max:  {np.max(deoxy_aucs):.5f}")
    
    results_text.append("")
    results_text.append("=" * 80)
    results_text.append("END OF ANALYSIS")
    results_text.append("=" * 80)

    print("=" * 50)
    print(f"AUC Summary with {goldenstandard} as the Golden Standard:")
    print("-"* 50)
    print("Oxy Analyses:")
    for fpr, tpr, color, name, roc_auc in all_oxy_results:
        print(f"{name}: AUC = {roc_auc:.5f}")
    print("=" * 50)
    print("Deoxy Analyses:")
    for fpr, tpr, color, name, roc_auc in all_deoxy_results:
        print(f"{name}: AUC = {roc_auc:.5f}")
    print("=" * 50)
    print("End of AUC Summary")
    print("\n\n")

    output_folder = r"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset\8_PerformanceEvaluation\ROC_fOLD"
    clean_name = goldenstandard.replace(' ', '_').replace('&', 'and')
    filename = f"ROC_Curves_{clean_name}.png"

    filepath = os.path.join(output_folder, filename)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(filepath, dpi = 300)
    plt.show()
    # Save the text results
    filename_txt = f"ROC_Results_{clean_name}.txt"
    filepath_txt = os.path.join(output_folder, filename_txt)
    with open(filepath_txt, 'w', encoding='utf-8') as f:
        f.write("\n".join(results_text))
    print(f"ROC curves and results saved to {filepath} and {filepath_txt}")

    # Print AUC for each analysis

def check_if_short_channels(df):
    """Check if the channel names are short (S##-D##). 
    Return 
        True if found
        or 
        False if not found
    """
    short_channels = df['Channel'].apply(lambda x: int(extract_d_value(x)) > 28)
    if short_channels.any() == True:
        # print(f"Short channels found. Please check the data.")
        return True
    else:
        # print(f"No short channels found. Proceeding with the analysis.")
        return False

def fOLD_binary():
    """Read the fOLD excel file that contains the channels and their respective category, then binirize them based on motor areas"""
    read_fold = pd.read_excel(r"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset\MULPA fOLD areas\Channels to Brain Areas using fOLD.xlsx", sheet_name='CLEAN')
    channel_category = read_fold.loc[:, ['Pair Satori', 'Category']]
    channel_category['Binary'] = channel_category['Category'].apply(lambda x: 1 if x in ['primary motor & somatosensory', 'secondary somatosensory', 'premotor'] else 0)
    channel_category.columns = ['Channel', 'Category', 'Binary']
    # Drop the 'Category' column if not needed
    channel_category = channel_category.drop(columns=['Category'])
    # Sort the channels by S and D values
    channel_category['S'] = channel_category['Channel'].apply(lambda x: int(x.split('-')[0][1:]))  # Extract S value
    channel_category['D'] = channel_category['Channel'].apply(lambda x: int(x.split('-')[1][1:]))  # Extract D value
    channel_category = channel_category.sort_values(by=['S', 'D']).reset_index(drop=True)
    # Remove the S and D columns after sorting
    channel_category = channel_category.drop(columns=['S', 'D'])
    channel_category['Channel'] = channel_category['Channel'].str.strip()  # Remove any leading/trailing whitespace
    # Print the first few rows of the channel_category DataFrame
    # print("fOLD Binary Channel Categories:")
    # print("=" * 50)
    # print(channel_category.to_string())
    # quit()
    return channel_category

def check_channel_names_match(df, goldern_standard_df):
    """Check if the channel names match in the given dataframes.
        if channels match 
            return True
        if channels do not match
            return False
        Input:
            df: list of tuples (name, dataframe) where name is the name of the dataframe and dataframe is the dataframe itself
            goldern_standard_df: dataframe with the golden standard channel names
    """
    reference_channels = goldern_standard_df['Channel']
    # print(reference_channels.head())
    # # quit()
    x=True
    for dataframe in enumerate(df):
        # print(dataframe[1][1]['Channel'].head())
        current_channels = dataframe[1][1]['Channel']
        # print("-" * 50)
        # print(current_channels.head())
        for cur_chan, ref_chan in zip(current_channels, reference_channels):
            if not (cur_chan == ref_chan):
                # print(f"Channel name mismatch in {dataframe[1][0]}: {cur_chan} vs {ref_chan}")
                x = False
    return x

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

def clean_name_endpos(file, end):
    """Extract the name from the file name after the given start string.
    Return the name till the given string
    """
    pos = file.find(end)  # Find the position of the start string 
    name = file[:pos]
    # name = name.split('.')[0]  # Remove the file extension
    return name


### Import our data
## Motor task data for oxy and deoxy: as golden standard
motor_path = r"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset\3_Data_PrePro\02a_test_MotorAction_GroupGLM_correctSerialC\MultiStudy_GLM_Results_MSGLM_Results_CorrectSC.xlsx"
motor_task_oxy = pd.read_excel(motor_path, sheet_name='Oxy Contrast Results')
motor_task_deoxy = pd.read_excel(motor_path, sheet_name='Deoxy Contrast Results')

# Clean the data
motor_task_oxy = motor_task_oxy.iloc[:,[1,2,3]].T.reset_index(drop=True).T
motor_task_deoxy = motor_task_deoxy.iloc[:,[1,2,3]].T.reset_index(drop=True).T
columns=["Channel", "Effsize", "pvalue"]

motor_task_oxy.columns = columns
motor_task_deoxy.columns = columns
motor_task_oxy = motor_task_oxy[1:].reset_index(drop=True) # remove the first row bcs it has the column names
motor_task_deoxy = motor_task_deoxy[1:].reset_index(drop=True)

# check and remove short channel names from the motor task data
if check_if_short_channels(motor_task_oxy) == True:
    motor_task_oxy = remove_short_channel_names(motor_task_oxy)
if check_if_short_channels(motor_task_deoxy) == True:
    motor_task_deoxy = remove_short_channel_names(motor_task_deoxy)

# reset the index of the dataframes
motor_task_oxy = motor_task_oxy.reset_index(drop=True)
motor_task_deoxy = motor_task_deoxy.reset_index(drop=True)

# Perform FDR correction on the p-values of motor task oxy/deoxy
motor_task_oxy['FDR_pvalue'] = stats.false_discovery_control(motor_task_oxy['pvalue'].astype(float), method='bh')
motor_task_deoxy['FDR_pvalue'] = stats.false_discovery_control(motor_task_deoxy['pvalue'].astype(float), method='bh')
# print(motor_task_oxy.to_string())
# quit()

# Binarize the motor task activation map based on the p-value threshold
motor_task_oxy.loc[motor_task_oxy['FDR_pvalue'] <= 0.05, 'Effsize'] = 1  # Significant activation
motor_task_oxy.loc[motor_task_oxy['FDR_pvalue'] > 0.05, 'Effsize'] = 0  # No significant activation
motor_task_deoxy.loc[motor_task_deoxy['FDR_pvalue'] <= 0.05, 'Effsize'] = 1  # Significant activation
motor_task_deoxy.loc[motor_task_deoxy['FDR_pvalue'] > 0.05, 'Effsize'] = 0  # No significant activation


# Get the path with the ttest results from SBA, ICA
ttest_files_path = r"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset\8_PerformanceEvaluation"

# ttest handling of SBA results (GLM, GLM-Resp, Corr)
seed_hbo_ttest_files = []  # name, path
seed_hbr_ttest_files = []  # name, path
ica_files = []  # name, path
for file in os.listdir(ttest_files_path):
    # Select the SBA files first, because the handling is the same across methods
    if file.startswith('SBA'):
        # select the file contains hbo 
        # print(file)
        if 'Oxy' in file:
            # extract the file name after the _S##
            name = clean_name_endpos(file, '_Oxy').replace("_", " ")
            seed_hbo_ttest_files.append((f"RSFC: {name}", os.path.join(ttest_files_path, file)))
        elif 'Deoxy' in file:
            # extract the file name after the _S##
            name = clean_name_endpos(file, '_Deoxy').replace("_", " ")
            seed_hbr_ttest_files.append((f"RSFC: {name}", os.path.join(ttest_files_path, file)))
    if file.startswith('ICA'):
        name = file.replace("_", " ").replace(".xlsx", "")
        ica_files.append((f"RSFC: {name}", os.path.join(ttest_files_path, file)))

# All the SBA and ICA names and paths are collected, now we can read the data from the excel files and place them in dataframes
## HbO and HbR t-test values
dataframes_oxy = []      # list of tuples (name, dataframe)
dataframes_deoxy = []  # list of tuples (name, dataframe)

column_names = ['Channel', 'Effsize']

## Oxy excels
# sba ttest excels 
for name, file_path in seed_hbo_ttest_files:
    data = pd.read_excel(file_path)
    data = data.iloc[:,[0,6]]  # Select the relevant columns: Channel, cohensd (6) 
    #! changed to t values for testing, results are the same
    data.columns = column_names # Rename the columns
    # check if there are short channels and remove them if they are present
    if check_if_short_channels(data) == True:
        data = remove_short_channel_names(data)
        # print("true")
    # print(name)
    # print(data.head())
    dataframes_oxy.append((name, data))


# ICA excels
for name, file_path in ica_files:
    data = pd.read_excel(file_path, sheet_name='Oxy Group RSFC')
    data = data.iloc[:,[0,6]]
    # Check if the columns are correct
    # print(data.head())
    # quit()
    data.columns = column_names  # Rename the columns
    data["Channel"] = data["Channel"].str.replace('hbo', '').str.replace('_','-').str.strip()  # Clean the channel names to match the other dataframes
    if check_if_short_channels(data) == True:
        data = remove_short_channel_names(data)
    print(name)
    # print(data.head())
    dataframes_oxy.append((name, data))

# for name, dataframe in dataframes_oxy:
#     print(f"name: {name}, dataframe shape: {dataframe.shape}")
# quit()

## Deoxy Excels
# ttest excels
for name, file_path in seed_hbr_ttest_files:
    data = pd.read_excel(file_path)
    data = data.iloc[:,[0,6]]  # Select the relevant columns: Channel, cohensd
    data.columns = column_names # Rename the columns
    if check_if_short_channels(data) == True:
        data = remove_short_channel_names(data)
    # print(name)
    # print(data.head())
    dataframes_deoxy.append((name, data))

# ICA excels
for name, file_path in ica_files:
    data = pd.read_excel(file_path, sheet_name='Deoxy Group RSFC')
    data = data.iloc[:,[0,6]]
    # print(data.head())
    # quit()
    # Check if the columns are correct
    data.columns = column_names  # Rename the columns
    #! this is the most important part, we need to invert the Effsize values 
    # for the deoxy data, due to the nature of ICA and its arbitrary sign convention. 
    # For the Oxy data it was aligned properly, for the deoxy data it was in the opposite 
    # direction. This can be seen when plotting the ROC curves IF NOT FLIPPED. The deoxy curves
    # UNDERPERFORM on a level that is way below chance, so it is on 'purpose'.
    data['Effsize'] = -data['Effsize'].astype(float)  # Ensure Effsize is float for ROC calculation
    data["Channel"] = data["Channel"].str.replace('hbr', '').str.replace('_','-').str.strip()  # Clean the channel names to match the other dataframes
    if check_if_short_channels(data) == True:
        data = remove_short_channel_names(data)
    dataframes_deoxy.append((name, data))
    # print(name)
    # print(data.head())

# # Check if the channel names match in the dataframes with that of the golden standard
if not check_channel_names_match(dataframes_oxy, motor_task_oxy):
    print("Channel names do not match in the dataframes. Please check the data.\nExiting the script...")
    quit()
if not check_channel_names_match(dataframes_deoxy, motor_task_deoxy):
    print("Channel names do not match in the dataframes. Please check the data.\nExiting the script...")
    quit()

## the same with fold binary dataframe as the golden standard
fold_binary_df = fOLD_binary()  # Get the fOLD binary dataframe with the channel names and their categories
fold_binary_df.columns = ['Channel', 'Effsize'] # make it compatible for plotting
# print(fold_binary_df.head())
# quit()
if not check_channel_names_match(dataframes_oxy, fold_binary_df):
    print("Channel names do not match in the dataframes with fOLD binary dataframe. Please check the data.\nExiting the script...")
    quit()
if not check_channel_names_match(dataframes_deoxy, fold_binary_df):
    print("Channel names do not match in the dataframes with fOLD binary dataframe. Please check the data.\nExiting the script...")
    quit()
    
for i, (name, dataframe) in enumerate(dataframes_oxy):
    check_inf(dataframe, values='Effsize')  # Check for infinity values in the Effsize column
    if check_inf(dataframe, values='Effsize'):
        print(f"Correcting infinity values in {name}...")
        corrected_df = correct_inf(dataframe, values='Effsize')
        # Replace the tuple in the list with a new tuple containing the corrected dataframe
        dataframes_oxy[i] = (name, corrected_df)
for i, (name, dataframe) in enumerate(dataframes_deoxy):
    check_inf(dataframe, values='Effsize')  # Check for infinity values in the Effsize column
    if check_inf(dataframe, values='Effsize'):
        print(f"Correcting infinity values in {name}...")
        corrected_df = correct_inf(dataframe, values='Effsize')
        # Replace the tuple in the list with a new tuple containing the corrected dataframe
        dataframes_deoxy[i] = (name, corrected_df)


# print(fold_binary_df.head())
# print(motor_task_oxy.head())
corr, p_value = stats.pearsonr(motor_task_oxy['Effsize'].astype(float), fold_binary_df['Effsize'].astype(float))
print(f"OXY: Correlation between motor task Effsize and fOLD Binary Effsize: {corr:.3f}, p-value: {p_value:.3f}")
corr, p_value = stats.pearsonr(motor_task_deoxy['Effsize'].astype(float), fold_binary_df['Effsize'].astype(float))

print(f"DEOXY: Correlation between motor task Effsize and fOLD Binary Effsize: {corr:.3f}, p-value: {p_value:.3f}")
quit()


# ## Normalizing the dataframes to 0-1 range; but its not really necessary for ROC analysis. I am doing it to replicate Behboodi et al. 2018
dataframes_oxy = normalize_dataframes(dataframes_oxy)
dataframes_deoxy = normalize_dataframes(dataframes_deoxy)

for name, df in dataframes_oxy:
    print(name)
    print(df.head())
    print("-" * 50)


plot_multiple_roc_curves(
                        golden_oxy=motor_task_oxy,
                        golden_deoxy=motor_task_deoxy,
                        oxy=dataframes_oxy,
                        deoxy=dataframes_deoxy,
                        goldenstandard='Motor Task Activation Map'
                        )

plot_multiple_roc_curves(
                        golden_oxy=fold_binary_df,
                        golden_deoxy=fold_binary_df,
                        oxy=dataframes_oxy,
                        deoxy=dataframes_deoxy,
                        goldenstandard='fOLD Binary'
                        )
# # Save the figures


quit()
# playing around with plotly.
# Lets create a plotly figure with the best 5 curves for both oxy and deoxy in a heatmap style

# Helper function to get the best performers based on AUC scores
# This function will return the top 5 performers for both oxy and deoxy analyses
def get_best_performers(golden_oxy, golden_deoxy, oxy, deoxy, n_best=5):
    """
    Extract the best performing dataframes based on AUC scores
    
    Returns:
        dict: {
            'oxy_best': [(name, df, auc_score), ...],
            'deoxy_best': [(name, df, auc_score), ...]
        }
    """
    # Calculate AUC for oxy analyses
    y_true_oxy = golden_oxy['Effsize'].astype(float)
    oxy_results = []
    
    for name, df in oxy:
        y_scores = df['Effsize'].astype(float)
        fpr, tpr, _ = roc_curve(y_true_oxy, y_scores)
        roc_auc = auc(fpr, tpr)
        oxy_results.append((name, df, roc_auc))
    
    # Calculate AUC for deoxy analyses  
    y_true_deoxy = golden_deoxy['Effsize'].astype(float)
    deoxy_results = []
    
    for name, df in deoxy:
        y_scores = df['Effsize'].astype(float)
        fpr, tpr, _ = roc_curve(y_true_deoxy, y_scores)
        roc_auc = auc(fpr, tpr)
        deoxy_results.append((name, df, roc_auc))
    
    # Sort by AUC and get top performers
    oxy_best = sorted(oxy_results, key=lambda x: x[2], reverse=True)[:n_best]
    deoxy_best = sorted(deoxy_results, key=lambda x: x[2], reverse=True)[:n_best]
    
    return {
        'oxy_best': oxy_best,
        'deoxy_best': deoxy_best
    }


print("Extracting best performers for additional analysis...")
# Get best performers for Motor Task golden standard
best_motor = get_best_performers(
    golden_oxy=motor_task_oxy,
    golden_deoxy=motor_task_deoxy, 
    oxy=dataframes_oxy,
    deoxy=dataframes_deoxy,
    n_best=5
)

# Get best performers for fOLD golden standard
best_fold = get_best_performers(
    golden_oxy=fold_binary_df,
    golden_deoxy=fold_binary_df,
    oxy=dataframes_oxy, 
    deoxy=dataframes_deoxy,
    n_best=5
)

# Print summary
print("\n" + "="*60)
print("BEST PERFORMERS SUMMARY")
print("="*60)
print("\nMotor Task Golden Standard:")
print("Top 5 Oxy:")
for i, (name, df, auc) in enumerate(best_motor['oxy_best'], 1):
    print(f"{i}. {name}: AUC = {auc:.3f}")

print("\nTop 5 Deoxy:")
for i, (name, df, auc) in enumerate(best_motor['deoxy_best'], 1):
    print(f"{i}. {name}: AUC = {auc:.3f}")

print("\nfOLD Binary Golden Standard:")
print("Top 5 Oxy:")
for i, (name, df, auc) in enumerate(best_fold['oxy_best'], 1):
    print(f"{i}. {name}: AUC = {auc:.3f}")

print("\nTop 5 Deoxy:")
for i, (name, df, auc) in enumerate(best_fold['deoxy_best'], 1):
    print(f"{i}. {name}: AUC = {auc:.3f}")
