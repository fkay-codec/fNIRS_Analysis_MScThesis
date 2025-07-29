"""
GROUP-LEVEL ICA-BASED RESTING-STATE FUNCTIONAL CONNECTIVITY (RSFC) ANALYSIS

This script is a continuation of the 'XXX.py'.

Short explanation:
    - The INPUT will be the results from the ICA analysis per individual.
        - REMINDER: per subject there are multiple runs of the FastICA algorithm with different
        random seeds anchored at 2025+i
    - For each individual per ICA run the IC that correlates highest with 
    the group-GLM motor task will be selected, sign aligned and stored into a 
    dataframe with Best_ICs_per_run_df.
    - Then for each subject the Best_ICs_per_run_df will averaged using the 
    meadian across all runs per channel; resulting in a single_IC_per_subject_df.
    - Then the group RSFC will be calculated by performing a one-sample t-test of these dataframes

LONG DESCRIPTION:
This script performs group-level analysis of resting-state functional 
connectivity using Independent Component Analysis (ICA) results. It identifies
motor-related independent components for each subject and computes group 
statistics to determine which brain regions shows connectivity
patterns across the group.

METHODOLOGY:
1. Subject-level IC Selection:
    - For each subject, for each run, correlate all ICs with group-level motor task T-values
    - For each subject, for each run, select the IC with the highest correlation (lowest p-value)
    - This will result in n_runs ICs per subject, where n_runs is the number of runs per subject
    - Sign-align the selected ICs with the group motor reference map
    - This identifies the most motor-relevant ICs for each individual grouped into one dataframe per subject
    - Then take the median of the ICs across all runs per subject
    - This ensures that the resultant median IC is stable across runs
    - Save the best ICs per subject in a dataframe

2. Group-level Statistical Analysis:
    - Now per subject we have a single IC that is most motor-related
    - perform group level anallysis across subjects with a one-sample t-test (H0: mean = 0)
    - Calculate:
        - T values for each channel
        - P values for each channel
        - Bonferroni/FDR correction for multiple comparisons across subjects
        - Mean and Standard Deviation for each channel
        - Effect sizes (Cohen's d) for each channel
    - Save the results in an Excel file

INPUT FILES:
- Per subject, per run: Excel files with IC spatial maps for both HbO and HbR
- Motor task GLM group results (Excel file): T-values for HbO and HbR per channels

OUTPUT FILES:
- Per subject: Excel files with the best ICs per run (HbO and HbR)
- Group-level analysis results: Excel file with T-values, P-values, FDR corrected P-values, Mean, Standard Deviation, and Effect sizes (Cohen's d) for each channel


QUALITY CONTROL:
- Channel name validation between IC data and motor reference
    - Ensure that the order of channels is the same in both datasets
    !if not, the script doesn't handle this yet and it will exit with an error message
    NOTE: If it exits then debugging is needed, not implemented because it is working for now.
- Short-distance channel removal (> 28)
    NOTE: removing short channels affects Satori SDKey. Take note that when removed, and wanting to create CMP files then the SDKey should include short channel names. There is a solution for this in [Folder ICA]:#a_create_cmp
- sign alignment of ICs with motor reference map
    - Ensures that the ICs are aligned with the expected motor task activation pattern


CONTINUATION OF THE PREVIOUS SCRIPT'S PSEUDO-CODE:
--------------------------------------------------
Part 1 of the code: FastICA with multiple runs
    1. for each subject run the ICA multiple times (N times) with a random seed 
    with fixed tolerance 
        Note: for reproducability and stability the random seed can have a specific starting point and then +i for each run; to validate the procedure pick different starting points and compare results; or just track the random seed generated
    2. store the resulting ICs and their spatial maps in a dataframe: each_run_ICs_df
    ! THIS SCRIPT STARTS FROM HERE
    3. for each IC calculate the spatial correlation with the motor reference map
    4. find the IC with the highest correlation
        a. perform sign alignment with the reference map
        b. store it in a new dataframe: e.g., best_ICs_df 
    5. save the best_ICs_df to an excel file with the subject ID in the filename

Output: Now you have the best_ICs per subject alligned and you have to decide how to proceed 
with IC selection.
--------------------------------
Part 2 of the code: IC selection
Part A: Use the best ICs from each subject across all runs and perform a group analysis
    1. load the best_ICs_df (name, dataframe)
    2. for each name (subject ID) in the dataframe:
        a. correlate the ICs with the group-level motor reference map
        b. select the IC with the highest correlation
        c. store the selected IC in a new dataframe: selected_ICs_df (name, dataframe)
            Note: check if sign is aligned with the reference map; if it isnt it will lead to a logical bug and the results are invalid
        d. save the selected_ICs_df to an excel file with the subject ID in the filename
        e. perform the group analysis on the selected ICs
        f. save the group analysis results to an excel file with the subject ID in the filename
Part B: Take the median from the best ICs for each subject and perform a group analysis
    1. load the best_ICs_df (name, dataframe) per run
    2. for each name (subject ID) in the dataframe:
        a. correlate the ICs across all runs to see if they are stable
            Note: before doing that check if IC sign is aligned
        b. if they are stable, average the ICs across all runs (they should be stable, if not, then you need to check the procedure)
        c. store the median IC in a new dataframe: averaged_ICs_df (name, dataframe)
        d. save the averaged_ICs_df to an excel file with the subject ID in the filename
    3. perform the group analysis across subjects on the averaged ICs
    4. save the group analysis results to an excel file and you are golden

    

NOTES: median is calculated correctly; ttest is performed correctly (cohend etc)

AUTHOR: Foivos Kotsogiannis
DATE: 17/07/2025
"""



import os
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import sys
sys.path.append(r"c:\Users\foivo\Documents\Python\Scripts")
import select_folder as sf

def check_channel_names_match(df, goldern_standard_df):
    """Check if the channel names match in the given dataframes.
        if channels match 
            return True
        if channels do not match
            return False
        Input:
            df: a dataframe with the IC spatial maps
            goldern_standard_df: dataframe with the motor task t-values
    """
    # reference_channels = goldern_standard_df['Channel']
    # print(reference_channels.head())
    # # quit()
    x=True
    for channel_df, channel_golden in zip(df['Channel'], goldern_standard_df['Channel']):
        if not (channel_df == channel_golden):
            # print(f"Channel name mismatch in {df.name}: {channel_df} vs {channel_golden}")
            x = False
    return x

def extract_d_value(channel):
    channel = channel.replace(" hbo", "").replace(" hbr", "")  # Remove the suffix to isolate the d value
    channel = channel.split('_')[-1]  # Extract the last part after the underscore, which is the: D##

    return channel[1:] # Extract the last part after the underscore, which is the d value

def clean_motor_df(df, heme_type=str):
    """Clean the motor task t-values DataFrame based on the heme type (HbO or HbR).
    Args:
        df (pd.DataFrame): The DataFrame containing the motor task t-values.
        heme_type (str): The type of heme ('hbo' for HbO, 'hbr' for HbR).
    """
    df = df.iloc[1:, [1, 2]]  # channel names in column B; t values in column C
    df.columns = ["Channel", "tvalue"]
    if heme_type == 'hbo':
        df["Channel"] = df["Channel"].str.replace("-", "_") + " hbo"  # do that to match IC_df, IMPORTANT: without this line extract_d_value will not work
    else:
        df["Channel"] = df["Channel"].str.replace("-", "_") + " hbr"
    df = df[df["Channel"].apply(lambda x: int(extract_d_value(x)) <= 28)]  # Remove short channels
    df["tvalue"] = df["tvalue"].astype(float)  # make them float64
    return df  

def perform_ttest_and_return_df(df):
    channel_names = []
    t_values = []
    p_values = []
    mean_values = []
    adjusted_p_values = []
    std_devs = []
    cohen_ds = []
    for i, row in df.iterrows():
        channel_name = row['Channel']       
        # extract the values for the channel
        values = pd.to_numeric(row[1:], errors='coerce')    # Skip the first column which is 'Channel'
        mean = values.mean()                    # Calculate the mean of the values
        std_dev = np.std(values, ddof=1)        # Calculate the SD of the values
        t_stat, p_val = stats.ttest_1samp(values, 0)        # Perform one-sample t-test (null hypothesis: z-value = 0; no activity across subjects for this channel)
        cohen_d = (mean - 0) / std_dev          # Calculate Cohen's d
        
        # Store the results
        channel_names.append(channel_name)
        t_values.append(t_stat)
        p_values.append(p_val)
        std_devs.append(std_dev)
        mean_values.append(mean)
        cohen_ds.append(cohen_d)
    # perform FDR correction for multiple comparisons across subjects
    adjusted_p_values = stats.false_discovery_control(p_values, method='bh')  # Use Benjamini-Hochberg method
    results_df = pd.DataFrame({
        "Channel": channel_names,
        "T-Value": t_values,
        "Raw P-Value": p_values,
        "FDR Adjusted P-Value": adjusted_p_values,
        "Standard Deviation": std_devs,
        "Mean Value": mean_values,
        "Cohen's d": cohen_ds
    })
    return results_df

def find_best_ic(df, reference_df):
    """"
    Find the best IC based on the highest correlation with the reference map.
    Return a DataFrame with the best IC values and the label of the best IC.
    """
    correlations_df = pd.DataFrame()
    for ic_column in df.columns[1:]:
        ic_values = df[ic_column].values
        correlation, p_value = stats.pearsonr(ic_values, reference_df['tvalue'].values)
        new_row = pd.DataFrame({
            "IC": [ic_column],
            "Correlation": [correlation],
            "P-value": [p_value]  
        })
        correlations_df = pd.concat([correlations_df, new_row], ignore_index=True)
    correlations_df = correlations_df.sort_values(by='P-value')    # Sort by p value and select the IC with the highest correlation
    # reset the index of the dataframe
    correlations_df.reset_index(drop=True, inplace=True)
    # print(correlations_df.to_string())
    best_ic_label = correlations_df.iloc[0]['IC']  # Get the label of the IC with the highest correlation
    best_ic_values = df[best_ic_label].astype(float)  # Get the values of the best IC
    # print(best_ic_label)
    return pd.DataFrame({best_ic_label: best_ic_values})

def remove_short_channel_names(df):
    """Remove short channel names from the dataframe and return it"""
    df = df[df['Channel'].apply(lambda x: int(extract_d_value(x)) <= 28)]
    return df

def check_sign_alignment(df, reference_df):
    """Sanity check to ensure that the ICs are sign aligned with the reference map."""
    for ic_column in df.columns[1:]:
        ic_values = df[ic_column].values
        correlation, p_value = stats.pearsonr(ic_values, reference_df['tvalue'].values)
        if correlation < 0:
            print(f"Warning: The IC {ic_column} is not sign aligned with the motor task reference map. Please check the data.")
            quit()

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

def find_median_ic(df):
    """Find the median IC across all runs for each subject."""
    median_ic = df.iloc[:, 1:].median(axis=1)
    # print(df.columns.values[1])
    pos = df.columns.values[1].find("_Run")
    clean_name = df.columns.values[1][:pos] + "_Median" # Clean the name of the subject by removing the run number
    # print(clean_name)
    median_ic_df = pd.DataFrame({clean_name: median_ic})
    return median_ic_df

# Extract the motor task t values for HbO and HbR from the group GLM results
# oxy & deoxy
motor_file_path = r"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset\3_Data_PrePro\02a_test_MotorAction_GroupGLM_correctSerialC\MultiStudy_GLM_Results_MSGLM_Results_CorrectSC.xlsx"
motor_oxy_df = pd.read_excel(motor_file_path ,sheet_name='Oxy Contrast Results')
motor_deoxy_df = pd.read_excel(motor_file_path, sheet_name='Deoxy Contrast Results')
motor_oxy_df = clean_motor_df(motor_oxy_df, 'hbo')  # Clean the motor task t-values DataFrame
motor_deoxy_df = clean_motor_df(motor_deoxy_df, 'hbr')  # Clean the motor task t-values DataFrame for deoxy
# print(motor_oxy_df.to_string())  # Print the first few rows of the motor task t-values DataFrame
# quit()
# this is a sanity check 
check_if_short_channels(motor_oxy_df)  # Check if there are short channels in the motor task t-values DataFrame
check_if_short_channels(motor_deoxy_df)  # Check if there are short channels in the motor task t-values DataFrame for deoxy

# Reset the index of the dataframes
motor_oxy_df.reset_index(drop=True, inplace=True)
motor_deoxy_df.reset_index(drop=True, inplace=True)

parent_input_folder = r"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset\7_ICA\1_FastICA_SM_Seed2025_Runs50"
# sf.select_folder_path("Select the parent folder with all ICA folders you need")

# not used because it will create a ton of files 2000 files in total... bcs its 22subjects x 50 runs x 2 functions
# # In this folder we will store folders per subject that will contain the correlations per run
# output_folder_per_subject_per_run_correlations = os.path.join(os.path.dirname(parent_input_folder), "6_ICA_Correlations_PerRun")
# if not os.path.exists(output_folder_per_subject_per_run_correlations):
#     os.makedirs(output_folder_per_subject_per_run_correlations)
#     print(f"Created output folder: {output_folder_per_subject_per_run_correlations}")

# in this folder we will store for each subject an excel that will contain the best ICs per run
output_folder_per_subject_grouped_best_ics = os.path.join(os.path.dirname(parent_input_folder), "6_AllRunsBestICs_PerSubject")
if not os.path.exists(output_folder_per_subject_grouped_best_ics):
    os.makedirs(output_folder_per_subject_grouped_best_ics)
    print(f"Created output folder: {output_folder_per_subject_grouped_best_ics}")

# due to sanity check we can proceed with these dataframes, they are clean and ready to use
# create a dataframe to store: for all subjects, the best IC across all runs
# per function
# logcosh
all_subj_best_ic_across_runs_oxy_logcosh = pd.DataFrame()
all_subj_best_ic_across_runs_oxy_logcosh["Channel"] = motor_oxy_df['Channel'].values  # Add the channel names to the DataFrame
all_subj_best_ic_across_runs_deoxy_logcosh = pd.DataFrame()
all_subj_best_ic_across_runs_deoxy_logcosh["Channel"] = motor_deoxy_df['Channel'].values  # Add the channel names to the DataFrame
# skew
all_subj_best_ic_across_runs_oxy_skew = pd.DataFrame()
all_subj_best_ic_across_runs_oxy_skew["Channel"] = motor_oxy_df['Channel'].values  # Add the channel names to the DataFrame
all_subj_best_ic_across_runs_deoxy_skew = pd.DataFrame()
all_subj_best_ic_across_runs_deoxy_skew["Channel"] = motor_deoxy_df['Channel'].values  # Add the channel names to the DataFrame
# associated folder for this output: I will store both dataframes in the same folder
output_folder_ic_selection_highest_corr_across_runs = os.path.join(os.path.dirname(parent_input_folder), "7a_IC_HighestCorr_AcrossRuns")
if not os.path.exists(output_folder_ic_selection_highest_corr_across_runs):
    os.makedirs(output_folder_ic_selection_highest_corr_across_runs)
    print(f"Created output folder: {output_folder_ic_selection_highest_corr_across_runs}")

# create a dataframe to store for all subjects, the median IC across all runs per function
# logcosh
all_subj_median_ic_across_runs_oxy_logcosh = pd.DataFrame()
all_subj_median_ic_across_runs_oxy_logcosh["Channel"] = motor_oxy_df['Channel'].values  # Add the channel names to the DataFrame
all_subj_median_ic_across_runs_deoxy_logcosh = pd.DataFrame()
all_subj_median_ic_across_runs_deoxy_logcosh["Channel"] = motor_deoxy_df['Channel'].values  # Add the channel names to the DataFrame
# skew
all_subj_median_ic_across_runs_oxy_skew = pd.DataFrame()
all_subj_median_ic_across_runs_oxy_skew["Channel"] = motor_oxy_df['Channel'].values  # Add the channel names to the DataFrame
all_subj_median_ic_across_runs_deoxy_skew = pd.DataFrame()
all_subj_median_ic_across_runs_deoxy_skew["Channel"] = motor_deoxy_df['Channel'].values  # Add the channel names to the DataFrame
# associated folder for this output: i will store both dataframes in the same folder
output_folder_ic_selection_median_across_runs = os.path.join(os.path.dirname(parent_input_folder), "7b_IC_Median_AcrossRuns")
if not os.path.exists(output_folder_ic_selection_median_across_runs):
    os.makedirs(output_folder_ic_selection_median_across_runs)
    print(f"Created output folder: {output_folder_ic_selection_median_across_runs}")

for folder in os.listdir(parent_input_folder):
    # we access each folder in the parent_input_folder that contains per subject folders that inside are the n_runs excels with the results of ICA
    # access each folder
    
    # create an empty dataframe to store the best ICs per run, for each subject
    all_best_subj_ic_oxy = pd.DataFrame()
    all_best_subj_ic_oxy["Channel"] = motor_oxy_df['Channel'].values  # Add the channel names to the DataFrame 

    all_best_subj_ic_deoxy = pd.DataFrame()
    all_best_subj_ic_deoxy["Channel"] = motor_deoxy_df['Channel'].values  # Add the channel names to the DataFrame

    for file in os.listdir(os.path.join(parent_input_folder, folder)):
        # now we are in each subject folder and we can access all the files: n_runs number of files
        oxy_data = pd.read_excel(os.path.join(parent_input_folder, folder, file), sheet_name='HbO Spatial Maps')
        deoxy_data = pd.read_excel(os.path.join(parent_input_folder, folder, file), sheet_name='HbR Spatial Maps')
        # Sanitize my dataframes; renaming first column and removing short channels
        oxy_data.columns.values[0] = 'Channel'  # Rename first column to 'Channel'
        deoxy_data.columns.values[0] = 'Channel'  # Rename first column to 'Channel' bcs it is not compatible with the motor task columns
        if not check_channel_names_match(oxy_data, motor_oxy_df):
            oxy_data = remove_short_channel_names(oxy_data)
        if not check_channel_names_match(deoxy_data, motor_deoxy_df):
            deoxy_data = remove_short_channel_names(deoxy_data)
        # Reset the index of the dataframes
        oxy_data.reset_index(drop=True, inplace=True)
        deoxy_data.reset_index(drop=True, inplace=True)

        # Sanity check: if the channel order is not the same in oxy-motor, deoxy-motor; Raise an error, otherwise we cant proceed with the correlations
        if not check_channel_names_match(oxy_data, motor_oxy_df) or not check_channel_names_match(deoxy_data, motor_deoxy_df):
            raise ValueError(f"Channel names do not match in {file}. Please check the data.")
        
        """    
        We are ready to proceed with these parts of the pseudo-code:
            3. for each IC calculate the spatial correlation with the motor reference map
            4. find the IC with the highest correlation
                a. perform sign alignment with the reference map
                b. store it in a new dataframe: e.g., best_ICs_df 
            5. save the best_ICs_df to an excel file with the subject ID in the filename
        """
        # Calculate the correlation between the ICs and the motor task t-values
        # create an empty dataframe to store the correlation results of each IC with the motor task t-values, per run
        correlations_oxy = pd.DataFrame()
        correlations_deoxy = pd.DataFrame()

        # Oxy best ICs which are stored in all_best_subj_ic_oxy
        for ic_column in oxy_data.columns[1:]:
            ic_values =  oxy_data[ic_column].values
            correlation, p_value = stats.pearsonr(ic_values, motor_oxy_df['tvalue'].values)
            new_row = pd.DataFrame({
                "IC": [ic_column],
                "Correlation": [correlation],
                "P-value": [p_value]  
            })
            # print(new_row.head())
        
            correlations_oxy = pd.concat([correlations_oxy, new_row], ignore_index=True)
            # print(correlations_oxy.head())
            # quit()
        correlations_oxy = correlations_oxy.sort_values(by='P-value')    # Sort by p value and select the IC with the highest correlation
        best_subj_ic_oxy_label = correlations_oxy.iloc[0]['IC']
        best_subj_ic_oxy_cor_sign = correlations_oxy.iloc[0]['Correlation']  # Get the sign of the correlation
        best_subj_ic_oxy = oxy_data[best_subj_ic_oxy_label].astype(float)  # Get the values of the best IC

        if best_subj_ic_oxy_cor_sign < 0: 
            best_subj_ic_oxy = -best_subj_ic_oxy

        # now store the best IC of the subject to the best_ic_oxy_all dataframe
        # first clean the name of the subject
        clean_subj_name_oxy = file.replace("01_RestingState_", "").replace(".xlsx", "")
        clean_subj_name_oxy = clean_subj_name_oxy + f"_{best_subj_ic_oxy_label}"  # add the IC label to the name
        new_column_df = pd.DataFrame({clean_subj_name_oxy: best_subj_ic_oxy})  # Create a new column with the best IC values
        all_best_subj_ic_oxy = pd.concat([all_best_subj_ic_oxy, new_column_df], axis=1)  # Add the new column to the dataframe
        
        # Deoxy best ICs which are stored in all_best_subj_ic_deoxy
        for ic_column in deoxy_data.columns[1:]:
            ic_values = deoxy_data[ic_column].values
            correlation, p_value = stats.pearsonr(ic_values, motor_deoxy_df['tvalue'].values)
            new_row = pd.DataFrame({
                "IC": [ic_column],
                "Correlation": [correlation],
                "P-value": [p_value]  
            })
            correlations_deoxy = pd.concat([correlations_deoxy, new_row], ignore_index=True)
        correlations_deoxy = correlations_deoxy.sort_values(by='P-value')    # Sort by p value and select the IC with the highest correlation
        
        # Here you can store the correlation oxy deoxy results if you want to but it will create a tone of them
            # for context: it will create a file for each subject x for each run x for each function = 2000, just dont 
            # store store
            # store store
        
        best_subj_ic_deoxy_label = correlations_deoxy.iloc[0]['IC']
        best_subj_ic_deoxy_cor_sign = correlations_deoxy.iloc[0]['Correlation']  # Get the sign of the correlation
        best_subj_ic_deoxy = deoxy_data[best_subj_ic_deoxy_label].astype(float)  # Get the values of the best IC
        if best_subj_ic_deoxy_cor_sign < 0:
            best_subj_ic_deoxy = -best_subj_ic_deoxy
        # now store the best IC of the subject to the best_ic_deoxy_all dataframe
        clean_subj_name_deoxy = file.replace("01_RestingState_", "").replace(".xlsx", "")
        clean_subj_name_deoxy = clean_subj_name_deoxy + f"_{best_subj_ic_deoxy_label}"  # add the IC label to the name
        new_column_df_deoxy = pd.DataFrame({clean_subj_name_deoxy: best_subj_ic_deoxy})  # Create a new column with the best IC values
        all_best_subj_ic_deoxy = pd.concat([all_best_subj_ic_deoxy, new_column_df_deoxy], axis=1)  # Add the new column to the dataframe



    # print(all_best_subj_ic_oxy)
    # print(all_best_subj_ic_deoxy)
    # Verify if the dataframe is sign aligned with the motor task reference map once more before saving
    # if they are not this function will rais an error and exit the script
    check_sign_alignment(all_best_subj_ic_oxy, motor_oxy_df)
    check_sign_alignment(all_best_subj_ic_deoxy, motor_deoxy_df)

    # Save the best ICs per subject to an excel file
    # create a name for the output file
    file_name = folder.replace("01_RestingState_", "") + "_bestICs" # P01_01_RestingState_logcosh
    with pd.ExcelWriter(os.path.join(output_folder_per_subject_grouped_best_ics, f"{file_name}.xlsx")) as writer:
        all_best_subj_ic_oxy.to_excel(writer, sheet_name='Oxy Best ICs', index=False)
        all_best_subj_ic_deoxy.to_excel(writer, sheet_name='Deoxy Best ICs', index=False)
    
    # Continue with the next part of the code: 
    # Part 2 of the code: IC selection
        # Part A: Use the best ICs from each subject across all runs and perform a group analysis
        #     1. load the best_ICs_df (name, dataframe)
        #     2. for each name (subject ID) in the dataframe:
        #         a. correlate the ICs with the group-level motor reference map
        #         b. select the IC with the highest correlation
        #         c. store the selected IC in a new dataframe: selected_ICs_df (name, dataframe)
        #             Note: check if sign is aligned with the reference map; if it isnt it will lead to a logical bug and the results are invalid
        #         d. save the selected_ICs_df to an excel file with the subject ID in the filename
        #         e. perform the group analysis on the selected ICs
        #         f. save the group analysis results to an excel file with the subject ID in the filename
        # Part B: Take the median from the best ICs for each subject and perform a group analysis
        #     1. load the best_ICs_df (name, dataframe) per run
        #     2. for each name (subject ID) in the dataframe:
        #         a. correlate the ICs across all runs to see if they are stable
        #             Note: before doing that check if IC sign is aligned
        #         b. if they are stable, average the ICs across all runs (they should be stable, if not, then you need to check the procedure)
        #         c. store the median IC in a new dataframe: averaged_ICs_df (name, dataframe)
        #         d. save the averaged_ICs_df to an excel file with the subject ID in the filename
        #     3. perform the group analysis across subjects on the averaged ICs
        #     4. save the group analysis results to an excel file and you are golden


    # find the most motor related IC across all runs per subject and append it to a dataframe to have one dataframe for all
    # two dataframes will be created, one for logcosh and one for skew 
    if "logcosh" in folder:
        # print(folder)
        # across all runs append the best ICs to the all_subj_best_ic_across_runs_oxy_logcosh dataframe       
        all_subj_best_ic_across_runs_oxy_logcosh = pd.concat(
            [all_subj_best_ic_across_runs_oxy_logcosh, find_best_ic(all_best_subj_ic_oxy, motor_oxy_df)], axis=1
        )
        all_subj_best_ic_across_runs_deoxy_logcosh = pd.concat(
            [all_subj_best_ic_across_runs_deoxy_logcosh, find_best_ic(all_best_subj_ic_deoxy, motor_deoxy_df)], axis=1
        )
        # across all runs compute the median IC and append it to teh all_subj_median_ic_across_runs_oxy_logcosh dataframe
        all_subj_median_ic_across_runs_oxy_logcosh = pd.concat(
            [all_subj_median_ic_across_runs_oxy_logcosh, find_median_ic(all_best_subj_ic_oxy)], axis=1
        )
        all_subj_median_ic_across_runs_deoxy_logcosh = pd.concat(
            [all_subj_median_ic_across_runs_deoxy_logcosh, find_median_ic(all_best_subj_ic_deoxy)], axis=1
        )
        
    elif "skew" in folder:
        # print(folder)
        all_subj_best_ic_across_runs_oxy_skew = pd.concat(
            [all_subj_best_ic_across_runs_oxy_skew, find_best_ic(all_best_subj_ic_oxy, motor_oxy_df)], axis=1
        )
        all_subj_best_ic_across_runs_deoxy_skew = pd.concat(
            [all_subj_best_ic_across_runs_deoxy_skew, find_best_ic(all_best_subj_ic_deoxy, motor_deoxy_df)], axis=1
        )
        # across all runs compute the median IC and append it to teh all_subj_median_ic_across_runs_oxy_skew dataframe
        all_subj_median_ic_across_runs_oxy_skew = pd.concat(
            [all_subj_median_ic_across_runs_oxy_skew, find_median_ic(all_best_subj_ic_oxy)], axis=1
        )
        all_subj_median_ic_across_runs_deoxy_skew = pd.concat(
            [all_subj_median_ic_across_runs_deoxy_skew, find_median_ic(all_best_subj_ic_deoxy)], axis=1
        )


# Now we are going to perform the group analysis across subjects on the best ICs based on Correlation and the best ICs based on the median across runs
# first on the dataframe: all_subj_best_ic_across_runs_(oxy/deoxy)_(logcosh/skew)


group_RSFC_corr_oxy_logcosh= perform_ttest_and_return_df(all_subj_best_ic_across_runs_oxy_logcosh)
group_RSFC_corr_deoxy_logcosh = perform_ttest_and_return_df(all_subj_best_ic_across_runs_deoxy_logcosh)
group_RSFC_corr_oxy_skew = perform_ttest_and_return_df(all_subj_best_ic_across_runs_oxy_skew)
group_RSFC_corr_deoxy_skew = perform_ttest_and_return_df(all_subj_best_ic_across_runs_deoxy_skew)

group_RSFC_median_oxy_logcosh = perform_ttest_and_return_df(all_subj_median_ic_across_runs_oxy_logcosh)
group_RSFC_median_deoxy_logcosh = perform_ttest_and_return_df(all_subj_median_ic_across_runs_deoxy_logcosh)
group_RSFC_median_oxy_skew = perform_ttest_and_return_df(all_subj_median_ic_across_runs_oxy_skew)
group_RSFC_median_deoxy_skew = perform_ttest_and_return_df(all_subj_median_ic_across_runs_deoxy_skew)


# now everythin is set and ready to save the results in an excel file
# for logcosh
with pd.ExcelWriter(os.path.join(output_folder_ic_selection_highest_corr_across_runs, "IC_IndCorr_logcosh.xlsx")) as writer:
    all_subj_best_ic_across_runs_oxy_logcosh.to_excel(writer, sheet_name='Oxy Best ICs', index=False)
    all_subj_best_ic_across_runs_deoxy_logcosh.to_excel(writer, sheet_name='Deoxy Best ICs', index=False)

with pd.ExcelWriter(os.path.join(output_folder_ic_selection_highest_corr_across_runs, "Group_RSFC_Hcor_logcosh.xlsx")) as writer:
    group_RSFC_corr_oxy_logcosh.to_excel(writer, sheet_name='Oxy Group RSFC', index=False)
    group_RSFC_corr_deoxy_logcosh.to_excel(writer, sheet_name='Deoxy Group RSFC', index=False)

with pd.ExcelWriter(os.path.join(output_folder_ic_selection_median_across_runs, "IC_IndMed_logcosh.xlsx")) as writer:
    all_subj_median_ic_across_runs_oxy_logcosh.to_excel(writer, sheet_name='Oxy Median ICs', index=False)
    all_subj_median_ic_across_runs_deoxy_logcosh.to_excel(writer, sheet_name='Deoxy Median ICs', index=False)

with pd.ExcelWriter(os.path.join(output_folder_ic_selection_median_across_runs, "Group_RSFC_Med_logcosh.xlsx")) as writer:
    group_RSFC_median_oxy_logcosh.to_excel(writer, sheet_name='Oxy Group RSFC', index=False)
    group_RSFC_median_deoxy_logcosh.to_excel(writer, sheet_name='Deoxy Group RSFC', index=False)

# for skew
with pd.ExcelWriter(os.path.join(output_folder_ic_selection_highest_corr_across_runs, "IC_GrCorr_skew.xlsx")) as writer:
    all_subj_best_ic_across_runs_oxy_skew.to_excel(writer, sheet_name='Oxy Best ICs', index=False)
    all_subj_best_ic_across_runs_deoxy_skew.to_excel(writer, sheet_name='Deoxy Best ICs', index=False)

with pd.ExcelWriter(os.path.join(output_folder_ic_selection_highest_corr_across_runs, "Group_RSFC_Hcor_skew.xlsx")) as writer:
    group_RSFC_corr_oxy_skew.to_excel(writer, sheet_name='Oxy Group RSFC', index=False)
    group_RSFC_corr_deoxy_skew.to_excel(writer, sheet_name='Deoxy Group RSFC', index=False)

with pd.ExcelWriter(os.path.join(output_folder_ic_selection_median_across_runs, "IC_GrMed_skew.xlsx")) as writer:
    all_subj_median_ic_across_runs_oxy_skew.to_excel(writer, sheet_name='Oxy Median ICs', index=False)
    all_subj_median_ic_across_runs_deoxy_skew.to_excel(writer, sheet_name='Deoxy Median ICs', index=False)

with pd.ExcelWriter(os.path.join(output_folder_ic_selection_median_across_runs, "Group_RSFC_Med_skew.xlsx")) as writer:
    group_RSFC_median_oxy_skew.to_excel(writer, sheet_name='Oxy Group RSFC', index=False)
    group_RSFC_median_deoxy_skew.to_excel(writer, sheet_name='Deoxy Group RSFC', index=False)

print("All results saved successfully!")
print("You can now proceed with the next steps of your analysis.")