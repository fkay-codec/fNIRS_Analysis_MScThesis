"""
PURPOSE:
--------
Converts group-level ICA resting-state functional connectivity (RSFC) results from Excel format to Satori CMP (Color Map) files 
for 3D brain visualization of motor-related independent component spatial maps.

1. DATA EXTRACTION:
    - Reads ICA group analysis Excel files with statistical results
    - Extracts t-values (column 2), Cohen's d effect sizes (column 7), and FDR-corrected p-values (column 4)
    - Processes both HbO and HbR chromophore data separately

2. SIGN ALIGNMENT:
    - Correlates ICA T-values with motor task GLM reference maps for proper orientation
    - ICA is sign-agnostic; applies sign correction if correlation with motor reference is negative

3. SPATIAL MAPPING:
    - Maps processed ICA channels to complete SATORI 134-channel layout
    - Handles missing channels (short channels removed during ICA preprocessing)
    - Creates master dataframes with standardized channel indexing

4. ONLY FOR SCALED CMP: STATISTICAL THRESHOLDING
    - Applies FDR-corrected p-value threshold (p ≤ 0.05) and normalizes the T-values from -1 to +1 by dividing with absolute max value, so the relative differences are maintained
    - Sets non-significant connectivity patterns to zero for cleaner visualization

5. CMP FILE GENERATION:
   Creates three separate brain maps per chromophore per input file:
    * Raw T-statistic map: Original statistical values for threshold-based analysis
    * Cohen's d map: Effect size magnitude for connectivity strength assessment  
    * Scaled T-statistic map: FDR-thresholded, normalized values optimized for visual comparison

INPUT:
- Excel files starting with "Group_" from ICA group analysis
- Must contain sheets: "Oxy Group RSFC" and "Deoxy Group RSFC"
- Required columns: Channel names (col 1), T-values (col 2), FDR P-values (col 4), Cohen's d (col 7)
- Channel format: "S##_D## hbo/hbr" (source-detector pairs with chromophore suffix)

OUTPUT:
For each input Excel file, generates 6 CMP files:
HbO Maps:
- "{filename}_tvalues_oxy.cmp": Raw T-statistic brain map for significance visualization
- "{filename}_CohensD_oxy.cmp": Effect size brain map for connectivity strength
- "{filename}_tvalues_scaled_oxy.cmp": FDR-thresholded, normalized T-statistic map

HbR Maps:  
- "{filename}_tvalues_deoxy.cmp": Raw T-statistic brain map for significance visualization
- "{filename}_CohensD_deoxy.cmp": Effect size brain map for connectivity strength
- "{filename}_tvalues_scaled_deoxy.cmp": FDR-thresholded, normalized T-statistic map
"""

import pandas as pd
import numpy as np
import os
import sys
sys.path.append(r"c:\Users\foivo\Documents\Python\Scripts")
import cmp as bvbabel
import select_folder as sf
import scipy.stats as stats

# # Comments on cmp files
# t-maps:
#         "TypeOfMap": "t-Map",
#         "dataType": "CC",
# beta-maps:
#         "TypeOfMap": "beta-Map",
#         "dataType": "CC",
# correlation maps:
#         "TypeOfMap": "r-Map",
#         "dataType": "r",



# Hardcoded Satori SDKey list because we have removed the short channels during processing with ICA and we cannot create accurate CMP files
SATORI_SDKEY =[
    "1-1",
    "1-2",
    "1-28",
    "1-29",
    "2-1",
    "2-2",
    "2-3",
    "2-4",
    "2-30",
    "3-1",
    "3-4",
    "3-31",
    "4-2",
    "4-3",
    "4-5",
    "4-26",
    "4-32",
    "5-3",
    "5-4",
    "5-6",
    "5-33",
    "6-4",
    "6-34",
    "7-3",
    "7-5",
    "7-6",
    "7-7",
    "7-35",
    "8-4",
    "8-6",
    "8-8",
    "8-36",
    "9-5",
    "9-7",
    "9-9",
    "9-23",
    "9-37",
    "10-6",
    "10-7",
    "10-8",
    "10-10",
    "10-38",
    "11-8",
    "11-11",
    "11-39",
    "12-7",
    "12-9",
    "12-10",
    "12-12",
    "12-40",
    "13-8",
    "13-10",
    "13-11",
    "13-13",
    "13-41",
    "14-9",
    "14-12",
    "14-19",
    "14-42",
    "15-10",
    "15-12",
    "15-13",
    "15-14",
    "15-43",
    "16-11",
    "16-13",
    "16-44",
    "17-12",
    "17-14",
    "17-15",
    "17-18",
    "17-19",
    "17-45",
    "18-15",
    "18-17",
    "18-18",
    "18-46",
    "19-14",
    "19-15",
    "19-16",
    "19-47",
    "20-15",
    "20-16",
    "20-17",
    "20-48",
    "21-18",
    "21-19",
    "21-20",
    "21-21",
    "21-49",
    "22-20",
    "22-22",
    "22-50",
    "23-9",
    "23-19",
    "23-21",
    "23-23",
    "23-51",
    "24-20",
    "24-21",
    "24-22",
    "24-24",
    "24-52",
    "25-21",
    "25-23",
    "25-24",
    "25-25",
    "25-53",
    "26-22",
    "26-24",
    "26-54",
    "27-5",
    "27-23",
    "27-25",
    "27-26",
    "27-55",
    "28-24",
    "28-25",
    "28-27",
    "28-56",
    "29-25",
    "29-26",
    "29-27",
    "29-57",
    "30-27",
    "30-58",
    "31-2",
    "31-26",
    "31-27",
    "31-28",
    "31-59",
    "32-27",
    "32-28",
    "32-60",
]

def custom_normalize(values):
    """
    Normalize the values to the range of -1 to 1.
    And preserve the original anchor 0.
        If the original value is 0, the rescaled value will also be 0.
    """
    values_array = np.array(values)
    min_val = np.min(values_array)
    max_val = np.max(values_array)
    if abs(min_val) > abs(max_val):
        # it is expected that it is smaller than the max positive
        scaled_div = abs(min_val)
    else:
        scaled_div = max_val
    values_array = values_array / scaled_div
    # print(scaled_div)
    # quit()

    return values_array.tolist() 

def handle_infinity_values(values):
    """
    Replace infinite values by replacing them with the maximum finite value in the dataset.
    """
    values_array = np.array(values)
    # print(values_array)
    # Create a mask for finite values
    finite_mask = np.isfinite(values_array)
    # print(finite_mask)
    # find the maximum finite value
    max_finite_value = np.max(values_array[finite_mask])
    # print(max_finite_value)

    # Replace infinite values with the maximum finite value
    values_array[~finite_mask] = max_finite_value
    # print(values_array)
    return values_array.tolist()

def clean_motor_df(df, heme_type=str):
    """Clean the motor task t-values DataFrame based on the heme type (HbO or HbR).
    Args:
        df (pd.DataFrame): The DataFrame containing the motor task t-values.
        heme_type (str): The type of heme ('hbo' for HbO, 'hbr' for HbR).
    """
    df = df.iloc[1:, [1, 2]]  # channel names in column B; t values in column C
    df.columns = ["Channel", "T-Value"]
    if heme_type == 'hbo':
        df["Channel"] = df["Channel"].str.replace("-", "_") + " hbo"  # do that to match IC_df, IMPORTANT: without this line extract_d_value will not work
    else:
        df["Channel"] = df["Channel"].str.replace("-", "_") + " hbr"
    df = df[df["Channel"].apply(lambda x: int(extract_d_value(x)) <= 28)]  # Remove short channels
    df["T-Value"] = df["T-Value"].astype(float)  # make them float64
    return df  

def extract_d_value(channel):
    channel = channel.replace(" hbo", "").replace(" hbr", "")  # Remove the suffix to isolate the d value
    channel = channel.split('_')[-1]  # Extract the last part after the underscore, which is the: D##

    return channel[1:] # Extract the last part after the underscore, which is the d value

# will be used for sign alignment to create accurate CMP files
motor_path = r"c:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset\3_Data_PrePro\02a_test_MotorAction_GroupGLM_correctSerialC\MultiStudy_GLM_Results_MSGLM_Results_CorrectSC.xlsx"

motor_oxy = pd.read_excel(motor_path, sheet_name="Oxy Contrast Results")
motor_oxy = clean_motor_df(motor_oxy, heme_type='hbo').reset_index(drop=True)

motor_deoxy = pd.read_excel(motor_path, sheet_name="Deoxy Contrast Results")
motor_deoxy = clean_motor_df(motor_deoxy, heme_type='hbr').reset_index(drop=True)


input_folder = sf.select_folder_path("Select folder with the ICA results")
if not input_folder:
    print("No folder selected. Exiting.\n")
    sys.exit()
# The output path, will be the same as the input folder


for file_name in os.listdir(input_folder):

    # Skip non-Excel files & files that dont have'ttest' suffix
    if not file_name.startswith("Group_") or not file_name.endswith(".xlsx"):
        print(f"Skipping file, not ttest result: {file_name}")
        continue
    print(f"Processing file: {file_name}")
    
    file_path = os.path.join(input_folder, file_name)

    data_oxy = pd.read_excel(file_path, sheet_name="Oxy Group RSFC")
    data_deoxy = pd.read_excel(file_path, sheet_name="Deoxy Group RSFC")
    # Extract the columns with names: 'Channel', 't-values', 'Cohen\'s d'
    data_oxy = data_oxy[['Channel', 'T-Value', "Cohen's d", "FDR Adjusted P-Value"]]
    data_deoxy = data_deoxy[['Channel', 'T-Value', "Cohen's d", "FDR Adjusted P-Value"]]
    # quit()

    # Before any more processing handle the sign
    # To create an accurate map, we need to see if the ICA weights are alligned with the motor_task_GLM
    # So basically the ICA is sign agnostic, and we need to align it to the direction of the correlation between the two.
    correlation_oxy, _ = stats.pearsonr(data_oxy['T-Value'], motor_oxy['T-Value'])
    if correlation_oxy < 0:
        print(f"Negative correlation detected: {correlation_oxy}. Flipping the sign of the t-values and Cohen's d for Oxy.")
        data_oxy['T-Value'] = -data_oxy['T-Value']
        data_oxy["Cohen's d"] = -data_oxy["Cohen's d"]
    correlation_deoxy, _ = stats.pearsonr(data_deoxy['T-Value'], motor_deoxy['T-Value'])
    if correlation_deoxy < 0:
        print(f"Negative correlation detected: {correlation_deoxy}. Flipping the sign of the t-values and Cohen's d for Deoxy.")
        data_deoxy['T-Value'] = -data_deoxy['T-Value']
        data_deoxy["Cohen's d"] = -data_deoxy["Cohen's d"]

    # Remove "S" and "D" from each value in the first column
    data_oxy['Channel'] = data_oxy['Channel'].str.replace("S", "").str.replace("D", "").str.replace(" hbo", "").str.replace("_","-")
    data_deoxy['Channel'] = data_deoxy['Channel'].str.replace("S", "").str.replace("D", "").str.replace(" hbr", "").str.replace("_","-")

    # Now use that SATORI_SDKEY as the main sdkey and create a new dataframe with the indexes of that one
    master_df_oxy = pd.DataFrame({
        "Channel": SATORI_SDKEY,
        "T-Value": [0.0] * len(SATORI_SDKEY),  # Initialize with dummy values
        "Cohen's d": [0.0] * len(SATORI_SDKEY),  # Initialize with dummy values
        "FDR P-Value": [1.0] * len(SATORI_SDKEY)  # Initialize with dummy values
    })
    master_df_deoxy = pd.DataFrame({
        "Channel": SATORI_SDKEY,
        "T-Value": [0.0] * len(SATORI_SDKEY),  # Initialize with dummy values
        "Cohen's d": [0.0] * len(SATORI_SDKEY),  # Initialize with dummy values
        "FDR P-Value": [1.0] * len(SATORI_SDKEY)  # Initialize with dummy values
    })
    for idx, row in data_oxy.iterrows():
        channel = row['Channel']
        
        if channel in SATORI_SDKEY:
            # Find the index in master_df where channel matches
            master_idx = master_df_oxy[master_df_oxy['Channel'] == channel].index[0]
            # Update the values at that index
            master_df_oxy.loc[master_idx, 'T-Value'] = row['T-Value']
            master_df_oxy.loc[master_idx, "Cohen's d"] = row["Cohen's d"]
            master_df_oxy.loc[master_idx, "FDR P-Value"] = row["FDR Adjusted P-Value"]

    for idx, row in data_deoxy.iterrows():
        channel = row['Channel']
        if channel in SATORI_SDKEY:
            # Find the index in master_df where channel matches
            master_idx = master_df_deoxy[master_df_deoxy['Channel'] == channel].index[0]
            # Update the values at that index
            master_df_deoxy.loc[master_idx, 'T-Value'] = row['T-Value']
            master_df_deoxy.loc[master_idx, "Cohen's d"] = row["Cohen's d"]
            master_df_deoxy.loc[master_idx, "FDR P-Value"] = row["FDR Adjusted P-Value"]
    
    

    
    master_df_oxy['T-Value Normalized'] = custom_normalize(master_df_oxy['T-Value'])
    # print(master_df_oxy.head(), "\n", min(master_df_oxy['T-Value']))
    master_df_deoxy['T-Value Normalized'] = custom_normalize(master_df_deoxy['T-Value'])

    # For the normalized t_values perform FDR thresholding
    # if FDR P-Value > 0.05, set the t-value to 0
    master_df_oxy['T-Value Normalized'] = master_df_oxy['T-Value Normalized'].where(master_df_oxy['FDR P-Value'] <= 0.05, 0)
    master_df_deoxy['T-Value Normalized'] = master_df_deoxy['T-Value Normalized'].where(master_df_deoxy['FDR P-Value'] <= 0.05, 0)

    base_name = os.path.splitext(file_name)[0]  # Remove the .xlsx extension


    header, maps = bvbabel.create_cmp(nr_maps=1)
    maps["map1"].update({
        "ContrastColor": "#ff0000",
        "MapName": f"{base_name}_tvalues_oxy",
        "DF1": 134,
        "DF2": 134,
        "NrOfChannels": 134,
        "NrOfDetectors": 60,
        "NrOfSources": 32,
        "NrOfWaveLength": 3,
        "SDKey": master_df_oxy['Channel'].tolist(),  # Use the SATORI_SDKEY
        "ShowCorrelationOrLag": 0,
        "ShowPosNegValues": 3,
        "ShowValuesAboveUpperThreshold": 1,
        "Threshold": 0,
        "TransparentColorFactor": 1,
        "TypeOfMap": "t-Map",
        "UpperThreshold": 10,
        "UseVMPColor": 0,
        "dataType": "CC",
        "rgbNegativeMaxValue": "#3b4cc0",
        "rgbNegativeMinValue": "#dddddd",
        "rgbPositiveMaxValue": "#b40426",
        "rgbPositiveMinValue": "#dddddd",
        "wl0_cmp": master_df_oxy['T-Value'].tolist(),
        "wl1_cmp": [0]*len(master_df_oxy['T-Value']),
        "wl2_cmp": [0]*len(master_df_oxy['T-Value'])
    })

    # Combine header and maps and write to file
    out_fname = os.path.join(input_folder, f"{base_name}_tvalues_oxy.cmp")
    
    bvbabel.write_cmp(out_fname, header, maps)
  
    # Cohen's D map
    header, maps = bvbabel.create_cmp(nr_maps=1)
    maps["map1"].update({
        "ContrastColor": "#ff0000",
        "MapName": f"{base_name}_cohens_d_oxy",
        "DF1": 134,
        "DF2": 134,
        "NrOfChannels": 134,
        "NrOfDetectors": 60,
        "NrOfSources": 32,
        "NrOfWaveLength": 3,
        "SDKey": master_df_oxy['Channel'].tolist(),  # Use the SATORI_SDKEY
        "ShowCorrelationOrLag": 0,
        "ShowPosNegValues": 3,
        "ShowValuesAboveUpperThreshold": 1,
        "Threshold": 0,
        "TransparentColorFactor": 1,
        "TypeOfMap": "t-Map",
        "UpperThreshold": 10, #! CUSTOM
        "UseVMPColor": 0,
        "dataType": "CC",
        "rgbNegativeMaxValue": "#3b4cc0",
        "rgbNegativeMinValue": "#dddddd",
        "rgbPositiveMaxValue": "#b40426",
        "rgbPositiveMinValue": "#dddddd",
        "wl0_cmp": master_df_oxy["Cohen's d"].tolist(),
        "wl1_cmp": [0]*len(master_df_oxy["Cohen's d"]),
        "wl2_cmp": [0]*len(master_df_oxy["Cohen's d"])
    })

    # Combine header and maps and write to file
    out_fname = os.path.join(input_folder, f"{base_name}_CohensD_oxy.cmp")

    bvbabel.write_cmp(out_fname, header, maps)

    # T map scaled
    header, maps = bvbabel.create_cmp(nr_maps=1)
    maps["map1"].update({
        "ContrastColor": "#ff0000",
        "MapName": f"{base_name}_tvalues_scaled_oxy",
        "DF1": 134,
        "DF2": 134,
        "NrOfChannels": 134,
        "NrOfDetectors": 60,
        "NrOfSources": 32,
        "NrOfWaveLength": 3,
        "SDKey": master_df_oxy['Channel'].tolist(),  # Use the SATORI_SDKEY
        "ShowCorrelationOrLag": 0,
        "ShowPosNegValues": 3,
        "ShowValuesAboveUpperThreshold": 1,
        "Threshold": 0,
        "TransparentColorFactor": 1,
        "TypeOfMap": "t-Map",
        "UpperThreshold": 1,
        "UseVMPColor": 0,
        "dataType": "CC",
        "rgbNegativeMaxValue": "#3b4cc0",
        "rgbNegativeMinValue": "#dddddd",
        "rgbPositiveMaxValue": "#b40426",
        "rgbPositiveMinValue": "#dddddd",
        "wl0_cmp": master_df_oxy['T-Value Normalized'].tolist(),
        "wl1_cmp": [0]*len(master_df_oxy['T-Value Normalized']),
        "wl2_cmp": [0]*len(master_df_oxy['T-Value Normalized'])
    })

    # Combine header and maps and write to file
    out_fname = os.path.join(input_folder, f"{base_name}_tvalues_scaled_oxy.cmp")
    
    bvbabel.write_cmp(out_fname, header, maps)

    # Deoxy maps
    header, maps = bvbabel.create_cmp(nr_maps=1)
    maps["map1"].update({
        "ContrastColor": "#ff0000",
        "MapName": f"{base_name}_tvalues_deoxy",
        "DF1": 134,
        "DF2": 134,
        "NrOfChannels": 134,
        "NrOfDetectors": 60,
        "NrOfSources": 32,
        "NrOfWaveLength": 3,
        "SDKey": master_df_deoxy['Channel'].tolist(),  # Use the SATORI_SDKEY
        "ShowCorrelationOrLag": 0,
        "ShowPosNegValues": 3,
        "ShowValuesAboveUpperThreshold": 1,
        "Threshold": 0,
        "TransparentColorFactor": 1,
        "TypeOfMap": "t-Map",
        "UpperThreshold": 10,
        "UseVMPColor": 0,
        "dataType": "CC",
        "rgbNegativeMaxValue": "#3b4cc0",
        "rgbNegativeMinValue": "#dddddd",
        "rgbPositiveMaxValue": "#b40426",
        "rgbPositiveMinValue": "#dddddd",
        "wl0_cmp": master_df_deoxy['T-Value'].tolist(),
        "wl1_cmp": [0]*len(master_df_deoxy['T-Value']),
        "wl2_cmp": [0]*len(master_df_deoxy['T-Value'])
    })

    # Combine header and maps and write to file
    out_fname = os.path.join(input_folder, f"{base_name}_tvalues_deoxy.cmp")
    
    bvbabel.write_cmp(out_fname, header, maps)
  
    # Cohen's D map
    header, maps = bvbabel.create_cmp(nr_maps=1)
    maps["map1"].update({
        "ContrastColor": "#ff0000",
        "MapName": f"{base_name}_cohens_d_deoxy",
        "DF1": 134,
        "DF2": 134,
        "NrOfChannels": 134,
        "NrOfDetectors": 60,
        "NrOfSources": 32,
        "NrOfWaveLength": 3,
        "SDKey": master_df_deoxy['Channel'].tolist(),  # Use the SATORI_SDKEY
        "ShowCorrelationOrLag": 0,
        "ShowPosNegValues": 3,
        "ShowValuesAboveUpperThreshold": 1,
        "Threshold": 0,
        "TransparentColorFactor": 1,
        "TypeOfMap": "t-Map",
        "UpperThreshold": 10, #! CUSTOM
        "UseVMPColor": 0,
        "dataType": "CC",
        "rgbNegativeMaxValue": "#3b4cc0",
        "rgbNegativeMinValue": "#dddddd",
        "rgbPositiveMaxValue": "#b40426",
        "rgbPositiveMinValue": "#dddddd",
        "wl0_cmp": master_df_deoxy["Cohen's d"].tolist(),
        "wl1_cmp": [0]*len(master_df_deoxy["Cohen's d"]),
        "wl2_cmp": [0]*len(master_df_deoxy["Cohen's d"])
    })

    # Combine header and maps and write to file
    out_fname = os.path.join(input_folder, f"{base_name}_CohensD_deoxy.cmp")

    bvbabel.write_cmp(out_fname, header, maps)

    # T map scaled
    header, maps = bvbabel.create_cmp(nr_maps=1)
    maps["map1"].update({
        "ContrastColor": "#ff0000",
        "MapName": f"{base_name}_tvalues_scaled_deoxy",
        "DF1": 134,
        "DF2": 134,
        "NrOfChannels": 134,
        "NrOfDetectors": 60,
        "NrOfSources": 32,
        "NrOfWaveLength": 3,
        "SDKey": master_df_deoxy['Channel'].tolist(),  # Use the SATORI_SDKEY
        "ShowCorrelationOrLag": 0,
        "ShowPosNegValues": 3,
        "ShowValuesAboveUpperThreshold": 1,
        "Threshold": 0,
        "TransparentColorFactor": 1,
        "TypeOfMap": "t-Map",
        "UpperThreshold": 1,
        "UseVMPColor": 0,
        "dataType": "CC",
        "rgbNegativeMaxValue": "#3b4cc0",
        "rgbNegativeMinValue": "#dddddd",
        "rgbPositiveMaxValue": "#b40426",
        "rgbPositiveMinValue": "#dddddd",
        "wl0_cmp": master_df_deoxy['T-Value Normalized'].tolist(),
        "wl1_cmp": [0]*len(master_df_deoxy['T-Value Normalized']),
        "wl2_cmp": [0]*len(master_df_deoxy['T-Value Normalized'])
    })

    # Combine header and maps and write to file
    out_fname = os.path.join(input_folder, f"{base_name}_tvalues_scaled_deoxy.cmp")
    
    bvbabel.write_cmp(out_fname, header, maps)