"""
fNIRS Resting-State PCA and FastICA Analysis - Spatial Map Generation


PURPOSE:
This script performs PCA and FastICA on preprocessed fNIRS resting-state data to generate Independent Component (IC) spatial maps for 
subsequent motor-RSFC analysis, following Zhang et al. (2010) methodology.
This script replicates partially Zhang et al. 2010, and makes several improvements:
    - Implements PCA for dimensionality reduction before FastICA
    - uses skew function as Zhang, and logcosh as alternative
    - because a fixed seed is deterministic, we run the FastICA multiple times with a random seed, while ensuring reproducability by using a fixed starting point and adding +i to it where i is the number of the runs
        - this way a common pitfall is avoided: using one random seed, one time gives one specific output, and when you run it again it gives another.
        - This is because the random seed is not fixed, so the random number generator is not initialized with the same seed every time.
        - Based on the above, when we run it multiple times with a random seed then the results become more stable
Later scripts further improve the methodology by:
    - Implementing a more robust IC selection that does not rely on visual inspection
    - Specifically, it selects the ICs that have the highest spatial correlation with the motor task GLM T-maps
    - then the ICs are aligned with the motor task glm maps (because ICA is sign agnotistic)
    - Best ICs across all runs are then saved in a dataframe for each subject
    - Then from the best ICs from each subject the median is taken to ensure stability of our results (and reproducability with a static starting point)
    - Finally the averaged ICs are used for group analysis, which is more robust than using the best ICs from each subject
==> All these improvements make ICA a robust and reliable method for fNIRS resting state analysis

METHODOLOGY OVERVIEW:
Based on Zhang et al. (2010) with the following pipeline:

1. DIMENSIONALITY REDUCTION (PCA)
---------------------------------
   - Apply PCA separately to HbO and HbR signals
   - Retain components explaining 99% of variance
   - Reduces computational load while preserving signal information
   
2. INDEPENDENT COMPONENT ANALYSIS (FastICA)
--------------------------------------------
   - Apply FastICA to PCA-reduced data
   - Number of ICs = Number of retained PCs
   - Extract spatially independent components from temporal signals
   
   FastICA Parameters (matching Zhang et al. 2010 MATLAB implementation):
   - Iterations: 10,000 (max_iter=10000)
   - Approach: "deflation" (algorithm="deflation") 
   - Initial value: "random" (random_state=None)
   - Nonlinearity: "skew" g(u)=u² (custom fun=skew)
   
    Note: Some MATLAB FastICA parameters not available in scikit-learn:
   - Step length: 0.00001 (no direct equivalent)
   - Fine-tune: "on" (not implemented)
   - Stabilization: "on" (not implemented)

3. SPATIAL MAP RECONSTRUCTION
-----------------------------
    - To recover spatial maps of each IC in the original channel space, the ICA mixing matrix 
    [the weights needed to reconstruct the PCA components; Each IC as a linear combination 
    of PCs] was multiplied by the PCA component matrix [the weights needed to reconstruct 
    the original channel contributions; Each PC as a linear combination of channels]. 
Script:
   - Recover IC spatial patterns in original channel space
        - Matrix multiplication: A = pca.components_.T @ ica_fit.mixing_
   - Z-score normalization of spatial maps for standardization
Code Explanation:
   - pca.components_.T: Channel contributions to each PC
   - ica_fit.mixing_: IC contributions from each PC
   - A: Final spatial maps (channels x ICs)

4. DATA ORGANIZATION
--------------------
   - Generate DataFrames with channels as rows, ICs as columns
   - Separate processing for HbO and HbR chromophores
   - Export to Excel with organized sheet structure

INPUT REQUIREMENTS:
    - folder '_SatPrep' suffix with snirf files 
    - SNIRF files must be preprocessed:
        - Converted to concentration changes (HbO/HbR)
        - Event-trimmed to uniform duration
        - Trimmed to reach steady state (first 20 sec last 20 seconds)
        - Detrended (1st order and 2nd order polynomial removal)
        - Bandpass filtered (0.01-0.2 Hz for resting-state frequencies)
OUTPUT FILES:
Excel files per subject: [SubjectID]_(...).xlsx
    - Two sheets per file:
        - 'HbO Spatial Maps': ICs for HbO channels
        - 'HbR Spatial Maps': ICs for HbR channels

SELECTION OF ICs:
! Not performed in this script; it is computationally intensive and done in a separate script
- Zhang et al. used visual inspection to select the ICs per subject 
    (ICs selection based on low frequency compoenents, that cover bilateral 
    motor areas for RSFC of motor areas)
    - There are better ways to do that, that enhance replicability and reproducibility:
        - Choose the IC that has the highest spatial correlation 
        with the spatial map of the motor areas (group-level GLM 
        analysis with motor event as predictor, yields this spatial map)
        - To ensure that the ICs dont have a high frequency content, bandpass the 
        data before PCA/ICA with a broad bandpass filter (in this case 0.01-0.2Hz)
--------------------------------

Best pracrices:
1.  Run ICA multiple times (e.g., 10-50 runs), each with a 
    different random_state. Use a fixed tolerance level (e.g., 1e-6)
    to ensure stability across runs.
2.  For each run, calculate the spatial correlation between
    each IC and your motor reference map.
3.  Track the best match across all runs, and log the following:
        - Max correlation coefficient,
        - Which IC from which run it came from,
        - The sign of the IC (since ICA is sign-agnostic).
You can then either:
a.  Pick the IC with the highest correlation across all runs (most common and simple).
        - align the sign of the IC with the reference map before saving. (this will be useful later for group analysis, otherwise its invalid)
b.  Or, average the best-matching ICs across runs (after sign alignment), if you're concerned about overfitting or instability.

Pseudo-code/steps for the above:
Part 1 of the code: FastICA with multiple runs
    1. for each subject run the ICA multiple times (N times) with a random seed 
    with fixed tolerance 
        Note: for reproducability and stability the random seed can have a specific starting point and then +i for each run; to validate the procedure pick different starting points and compare results; or just track the random seed generated
    2. store the resulting ICs and their spatial maps in a dataframe: each_run_ICs_df
    ! this script stops here, the next part is in a different script
    3. for each IC calculate the spatial correlation with the motor reference map
    4. find the IC with the highest correlation
        a. perform sign alignment with the reference map
        b. store it in a new dataframe: best_ICs_df 
    5. save the best_ICs_df to an excel file with the subject ID in the filename

Output: Now you have the best_ICs per subject alligned and you have to decide how to proceed with IC selection.
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
Part B: Tale the median from the best ICs for each subject and perform a group analysis
    1. load the best_ICs_df (name, dataframe) per run
    2. for each name (subject ID) in the dataframe:
        a. correlate the ICs across all runs to see if they are stable
            Note: before doing that check if IC sign is aligned
        b. if they are stable, average the ICs across all runs (they should be stable, if not, then you need to check the procedure)
        c. store the median IC in a new dataframe: averaged_ICs_df (name, dataframe)
        d. save the averaged_ICs_df to an excel file with the subject ID in the filename
    3. perform the group analysis across subjects on the averaged ICs
    4. save the group analysis results to an excel file and you are golden

NOTE: total running time for: 23 subjects x 50 runs x 2 functions = 2300 runs in 17.2hrs
NOTE: logcosh function is slower than skew
    
Author: Foivos Kotsogiannis
Date: 15/7/2025    
"""



import sys
sys.path.append(r"c:\Users\foivo\Documents\Python\Scripts")
import cmp as bvbabel

import time
import os
import mne
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA, PCA
from PySide6.QtWidgets import QApplication, QFileDialog

def select_folder_path(prompt):
    # Open dialog to select the folder of interest and return its path
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    # Open a dialog to select a folder 
    folder_path = QFileDialog.getExistingDirectory(None, prompt, r"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset")
    return folder_path
def channel_order_check(oxy_ic_data_df, deoxy_ic_data_df, motor_oxy_tvalues, motor_deoxy_tvalues, file):
    """ 
    Check if the channel names in the IC data match the motor task t-values, and if they match in order.
    If not, raise an error and exit the script.
    if the error occurs, then we need to build a debugging function to solve this issue.
    """
    # Check if the channel names in the IC data match the motor task t-values, and if they match in order
    for channel_oxy_ic, channel_oxy_motor in zip(oxy_ic_data_df['Channel'], motor_oxy_tvalues['Channel']):
        if channel_oxy_ic != channel_oxy_motor:
            print(f"    Exiting script due to channel name mismatch in {file}.")
            print("     If this happens we need to build a debuging function to solve this issue.")
            # if this happens we need to build a debuging function to solve this issue.
            quit()
    for channel_deoxy_ic, channel_deoxy_motor in zip(deoxy_ic_data_df['Channel'], motor_deoxy_tvalues['Channel']):
        if channel_deoxy_ic != channel_deoxy_motor:
            print(f"    Exiting script due to channel name mismatch in {file}.")
            print("     If this happens we need to build a debuging function to solve this issue.")
            quit()
    # print(f"Channel names match for HbO/HbR in {file}.")

def extract_d_value(channel):
    """Designed to extract the d value from the channel name. Only works for channels with the format: D##_hbo or D##_hbr.
    keep as is and use a custom variation of this if needed. IMPORTANT DONT CHANGE THIS FUNCTION. JUST CREATE A NEW ONE IF NEEDED.
    """
    channel = channel.replace(" hbo", "").replace(" hbr", "")  # Remove the suffix to isolate the d value
    channel = channel.split('_')[-1]  # Extract the last part after the underscore, which is the: D##

    return channel[1:] # Extract the last part after the underscore, which is the d value

def extract_clean_filename(filename):
    """
    Extract clean filename from start until 'RestingState' (inclusive).
    
    Examples:
    - P01_01_RestingState_TrimmedToEvents_OD_UnifDur_Detr_HP02_LP001_Trim.snirf → P01_01_RestingState
    - S04_01_RestingState_ProbOverflow_Manual_OD_TrimToEvents_UnifDur_Detr_HP02_LP001_Trim.snirf → S04_01_RestingState
    """
    # Remove file extension first
    base_name = os.path.splitext(filename)[0]
    
    # Find the position of "RestingState" 
    resting_state_pos = base_name.find("RestingState")
    
    if resting_state_pos != -1: # if "RestingState" is found
        # Extract from start to end of "RestingState"
        clean_name = base_name[:resting_state_pos + len("RestingState")]
        return clean_name
    else:
        print(f"'RestingState' not found in filename: {filename}")
        print("Using first 3 parts of the filename instead.")
        # Fallback: if "RestingState" not found, use first 3 parts
        parts = base_name.split('_')
        return '_'.join(parts[:3]) if len(parts) >= 3 else base_name

def skew(x):
    """Create custom skew function for FastICA to match MATLAB FastICA.
    Replicating: %'skew'g(u)=u^2 from FastICA_25/fastica.m
    Returns:
    --------
    tuple : (g, g_der)
        g : nonlinearity function g(u) = u^2
        g_der : derivative g'(u) = 2*u, averaged along last dimension
    """
    return x ** 2, (2 * x).mean(axis=-1)  

def detrend_polynomial(data):
    """Detrend data using polynomial fitting (2nd order)."""
    n_channels, n_samples = data.shape
    detrended_data = np.zeros_like(data, dtype=float)
    x = np.arange(n_samples)  # Time points
    for ch in range(n_channels):
        # Fit a 2nd order polynomial to the channel data
        coeffs = np.polyfit(x, data[ch], 2)
        # Evaluate the polynomial at the time points
        trend = np.polyval(coeffs, x)
        # Subtract the trend from the original data
        detrended_data[ch] = data[ch] - trend
    return detrended_data

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
    df["tvalue"] = df["tvalue"].astype(float)  # Extract only the t-values for correlation calculation as float64
    return df  

def myICA(function_name, function, n_runs, tolerance, starting_point):
    """Because the process is computationally intensive and takes a long time,
    we will run the PCA and FastICA with n_runs with a random seed, different funcitions and fixed tolerance
    
    Then Per subject save the resultsing IC spatial maps from each run of the FastICA algorithm in an excel in an excel file for each run per subject 
    """
    ###* Part 1 of the code: FastICA with multiple runs
    # 1. for each subject run the ICA multiple times (N times) with a random seed with fixed tolerance
    # 2. store the resulting ICs and their spatial maps in a dataframe: each_run_ICs_df
    # 3. for each IC calculate the spatial correlation with the motor reference map
    # 4. find the IC with the highest correlation
    #     a. perform sign alignment with the reference map
    #     b. store it in a new dataframe: best_ICs_df
    # 5. save the best_ICs_df to an excel file with the subject ID in the filename
    

    # input folder with preprocessed SNIRF files ready for PCA/ICA; has prefix: '_SatPrep'; the files are not detrended!
    input_folder = select_folder_path("Select the folder with preprocessed SNIRF files for PCA/ICA analysis")
    
    """Perform PCA and FastICA on preprocessed fNIRS data."""
    #! Create output folder based on input folder: we will see how to name it
    output_folder = os.path.join(os.path.dirname(input_folder), f"1_FastICA_SM_Seed{str(starting_point)}_Runs{n_runs}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    for file in os.listdir(input_folder):
        #! Custom because it crashed
        participant_number = file.split('_')[0]  # Extract participant number from filename
        participant_number = int(participant_number[1:])  # Remove first character and convert to int (e.g., 31)

        if function_name == 'logcosh' and participant_number < 31:
            print(f"skipping file {file} with func {function_name} beacuse its already processed and the program crashed")
            continue

        #! Custom because it crashed
        if file.endswith(".snirf"):
            print(f"PROCESSING: {file}\n")
            input_file = os.path.join(input_folder, file)
            raw = mne.io.read_raw_snirf(input_file, preload=True)
            clean_filename = extract_clean_filename(file)  # Extract clean filename for output
            print(f"\nParticipant: {clean_filename}\n")
            
            # Create an output folder for each subject
            output_folder_for_each_subject = os.path.join(output_folder, f"{clean_filename}_{function_name}")
            if not os.path.exists(output_folder_for_each_subject):
                os.makedirs(output_folder_for_each_subject)
                print(f"Created output folder for subject: {output_folder_for_each_subject}")
            
            #* 1. for each subject run the ICA multiple times (N times) with a random seed with fixed tolerance

            # extract the hbo and hbr signals from the raw data along with their channels
            # * HbO:
            hbo_picks = mne.pick_types(raw.info, fnirs='hbo')
            hbo_channel_names = [raw.ch_names[i] for i in hbo_picks]
            hbo_timeseries= raw.get_data(picks=hbo_picks)  # shape: (n_channels, n_samples)
            #! New addition; found a way to detrend using the secord order polynomial fit; maybe this is why i didnt have good ICA results before
            hbo_timeseries = detrend_polynomial(hbo_timeseries)  # Detrend the data (2nd order polynominal detrending)
            # store the data in dataframe
            hbo_df = pd.DataFrame(hbo_timeseries.T, columns = hbo_channel_names)   # Build a DataFrame: Columns are channels, rows are time points

            # --- PCA ---
            print("=" * 50)
            print("Performing PCA on HbO data...")
            pca_hbo = PCA(n_components=0.99, svd_solver='full') # apply PCA to the HbO data, with number of components that explain 99% of variance
            pca_hbo_fit = pca_hbo.fit(hbo_df)
            pca_hbo_transformed = pca_hbo.transform(hbo_df)
            print("PCA completed for HbO data.")
            
            # The same with HbR data:
            # * for HbR:
            hbr_picks = mne.pick_types(raw.info, fnirs='hbr')
            hbr_channel_names = [raw.ch_names[i] for i in hbr_picks]
            hbr_timeseries = raw.get_data(picks=hbr_picks) # shape: (n_channels, n_samples)
            #! Detrend the HbR data 2nd order
            hbr_timeseries = detrend_polynomial(hbr_timeseries)  # Detrend the data (2nd order polynominal detrending)          
            hbr_df = pd.DataFrame(hbr_timeseries.T, columns = hbr_channel_names)    # Build a DataFrame: Columns are channels, rows are time points
            
            print("=" * 50)
            print("Performing PCA on HbR data...")
            pca_hbr = PCA(n_components=0.99, svd_solver='full')
            pca_hbr_fit = pca_hbr.fit(hbr_df)
            pca_hbr_transformed = pca_hbr.transform(hbr_df)
            print("PCA completed for HbR data.")
                                   
            # --- FastICA ---
            """
            based on Zhang et al. 2010
            reduce the data dimensionality for each subject using PCA - retaining 99% of the variance to compute the number of components
            the reduced data is then used in the FastICA algorithm with the number of IC equal to the number of PC 
            FastICA algorithm parameters from Zhang: 
                number of iteration steps=10000, step length= 0.00001, approach=“deﬂation”, initial value=“random”, nonlinearity = “skew”, ﬁne-tune = “on”, and stabilization = “on”.

            random_state in scikit-learn

            - If random_state=None (default), the global numpy random generator is used, so results may differ each run.
            - If random_state is set to an integer (e.g., 0 or 42), the results will be reproducible: you will get the same output every time you run the code with the same data and parameters.

            For replication of Zhang et al. 2010, should set random_state to None
            """
            
            print("=" * 50)
            print("Performing FastICA on HbO & HbR data...")
            time2 = time.time()  # Start timer for processing time
            for i in range(n_runs):
                print(f"Run {i+1}/{n_runs} with function: {function_name} and tolerance: {tolerance:.0e}")
                print("=" * 50)
                print("Performing FastICA on HbO data...")
                ica_hbo = FastICA(
                    n_components=pca_hbo_transformed.shape[1],
                    algorithm="deflation",          # same as Zhang et al. 2010
                    max_iter=10000,                 # same as Zhang et al. 2010
                    fun=function,                   # 2 functions: 'logcosh' or 'skew'; skew is the same as Zhang et al. 2010
                    random_state=starting_point+i,  # perform ica with a stable starting point for reproducibility. Because using a fixed seed is deterministic, we will use a range of seeds and then take the median of all runs to increase stability        
                    tol = tolerance,                # tolerance selected is low to ensure convergence, but not too low to avoid excessive computation time
                ) 
                ica_hbo_fit = ica_hbo.fit(pca_hbo_transformed)                  # Fit FastICA to the reduced data
                ica_hbo_transformed = ica_hbo.transform(pca_hbo_transformed)    # Transform the filtered data into the independent components
                """
                pca.components_ : shape: (21, 134)
                    pca.components_ contains the principal component loadings 
                    these are the weights that define how each original channel contributes to each principal component
                ica_fit.mixing_ : ica_fit.mixing_ shape: (21, 21)
                    ica_fit.mixing_ is the mixing matrix from ICA.
                    it contains the weights that define how each independent component (IC) is formed from the original channels
                    how they combine to reconstruct the original data
                by multiplying the PCA componets and their loadings with the ICA mixing matrix, we can reconstruct the independent components (ICs) in terms of the original channels
                """
                spatial_maps_hbo = pca_hbo.components_.T @ ica_hbo_fit.mixing_  # spatial maps: the mixing matrix, where each row is a channel and each column is an IC; shape: (n_channels, n_ICs)
                spatial_maps_hbo = (spatial_maps_hbo - spatial_maps_hbo.mean(axis=0)) / spatial_maps_hbo.std(axis=0)    # Normalize the mixing matrix (z-score normalization)
                spatial_maps_hbo_df = pd.DataFrame(spatial_maps_hbo, columns=[f'IC{i+1}' for i in range(spatial_maps_hbo.shape[1])], index=hbo_channel_names)  # Build a DataFrame: Columns are ICs, rows are channels
                print("FastICA completed for HbO data.")
                print("=" * 50)
                print("Performing ICA on HbR data...")
                ica_hbr = FastICA(
                    n_components=pca_hbr_transformed.shape[1],
                    algorithm="deflation",          
                    fun=function,                   
                    random_state=starting_point+i,              
                    tol = tolerance,
                )
                ica_hbr_fit = ica_hbr.fit(pca_hbr_transformed)
                ica_hbr_transformed = ica_hbr.transform(pca_hbr_transformed)
                spatial_maps_hbr = pca_hbr.components_.T @ ica_hbr_fit.mixing_
                spatial_maps_hbr = (spatial_maps_hbr - spatial_maps_hbr.mean(axis=0)) / spatial_maps_hbr.std(axis=0)
                spatial_maps_hbr_df = pd.DataFrame(spatial_maps_hbr, columns=[f'IC{i+1}' for i in range(spatial_maps_hbr.shape[1])], index=hbr_channel_names)
                print(f"ICA completed for HbR data.")

                # 2. The spatial maps from each run are now saved in excel files in subject folders  
                excel_filename = f"{clean_filename}_{function_name}_Run{i+1}.xlsx"
                excel_file_path = os.path.join(output_folder_for_each_subject, excel_filename)
                with pd.ExcelWriter(excel_file_path) as writer:
                    spatial_maps_hbo_df.to_excel(writer, sheet_name='HbO Spatial Maps', index=True, header=True)
                    spatial_maps_hbr_df.to_excel(writer, sheet_name='HbR Spatial Maps', index=True, header=True)
            print(f"Completed FastICA for {clean_filename}")
            print(f"Files saved all files {output_folder_for_each_subject}\n")
            end2 = time.time()  # End timer for processing time
            print(f"Total time: {(end2 - time2)/60:.2f}min for participant {clean_filename} with {function_name} and n_runs: {n_runs}")
            # with vscode setting on
    print("=" * 50)
    print("All files processed successfully!")
    print(f"Output files saved to: {output_folder}")
    print("=" * 50)


start = time.time()

# ## Constants
start_seed = 2025 # this is the seed we are going to use for the fastICA algorithm as a starting seed. And then n_runs will add + i to it. Ex: n_runs = 50 then it will run the algorithm from 2025 to 2074. This is done for reproducability and stability of the results. Then afterwards I will test with n_runs =50 with a different starting point
n_runs = 50  # Number of ICA runs per subject
tole = 1e-6  # Tolerance for FastICA convergence
function = [['logcosh','logcosh'], ['skew', skew]]  # Nonlinearity function for FastICA


for i in range(len(function)):
    print("=" * 50)
    print(f"Running FastICA with function: {function[i][0]} with {n_runs} random seeds and tolerance {tole:.0e}\n")
    myICA(function_name=function[i][0], function=function[i][1], n_runs=n_runs, tolerance=tole, starting_point=start_seed)  # Call the myICA function with the current function name and function
    print(f"Completed FastICA with function: {function[i][0]}\n")
    print("=" * 50)

end = time.time()
print(f"Total processing time: {(end - start)/60:.2f} minutes")




