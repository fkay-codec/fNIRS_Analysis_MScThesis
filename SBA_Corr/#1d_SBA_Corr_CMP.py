"""
!SAME SCRIPT AS #3 ttest cmp.py from SBA GLM, but with handles infinity values
   - **Seed Channel Handling:**
      - Seed channels show perfect self-correlation (r = 1.0, std = 0)
      - Results in infinite t-statistics and Cohen's d values
      - Displayed as "inf" in output - mathematically correct representation
Here we assign the max value to the inf values, so that they can be visualized in the cmp files


PURPOSE:
--------
Converts group-level t-test results from Excel format to Satori CMP (Color Map) files 
for 3D brain visualization of seed-based connectivity patterns.

1. DATA EXTRACTION:
    - Reads t-test Excel files with statistical results from group analysis
    - Extracts t-values (column 2) and Cohen's d effect sizes (column 7)

2. CMP FILE GENERATION:
   - Creates two separate brain maps per input file:
    * T-statistic map: Shows statistical significance of connectivity
    * Cohen's d map: Shows effect size magnitude of connectivity strength

INPUT:
- Excel files ending with "_ttest.xlsx" from group statistical analysis
- Must contain: Channel names (col 1), T-values (col 2), Cohen's d (col 7)
- Channel format: "S##D##" (source-detector pairs)

OUTPUT:
For each input Excel file, generates:
- "{filename}_tvalues.cmp": T-statistic brain map for significance visualization
- "{filename}_CohensD.cmp": Effect size brain map for connectivity strength
- "{filename}_tvalues_scaled.cmp": Scaled T-statistic map for better visualization, non-significant channels assigned with 0 t and cohens d values. 
    - visualizing only significant channels, while they are scaled from -1 to +1 by dividing with the abs max value of the values. Ensuring that the 0 anchor is preserved and the relative differences are maintained.
"""
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(r"c:\Users\foivo\Documents\Python\Scripts")
import cmp as bvbabel
import select_folder as sf

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

input_folder = sf.select_folder_path("Select folder with the ttest results")
if not input_folder:
    print("No folder selected. Exiting.\n")
    sys.exit()
# The output path, will be the same as the input folder

for file_name in os.listdir(input_folder):

    # Skip non-Excel files & files that dont have'ttest' suffix
    if not file_name.endswith("ttest.xlsx"):
        print(f"Skipping non-Excel file or not ttest result: {file_name}")
        continue

    file_path = os.path.join(input_folder, file_name)

    data = pd.read_excel(file_path)

    # Extract the first column from the DataFrame
    first_column = data.iloc[:, 0]
    # print("First column values:", first_column)

    # Remove "S" and "D" from each value in the first column
    processed_sdkey = [key.replace("S", "").replace("D", "") for key in first_column]

    # Extract the second/seventh column tvalues/coehend values
    t_values = data.iloc[:, 1] #tvalues
    t_values = handle_infinity_values(t_values)  # Handle infinite values
    cohens_ds = data.iloc[:, 6] #cohends
    cohens_ds = handle_infinity_values(cohens_ds)  # Handle infinite values
    base_name = os.path.splitext(file_name)[0]  # Remove the .xlsx extension
    
    # for the third map we will repeat the process by creating another variable t_values_scaled
    # Rescale the t-values to -1 to +1 for better visualization
    t_values_scaled = data.iloc[:, 1]

    # Extract the p values and if p > 0.05 set the t-values
    fdr_p_values = data.iloc[:, 3] # FDR p-values
    t_values_scaled = t_values_scaled.where(fdr_p_values <= 0.05, 0)  # Set t-values to 0 where p > 0.05

    t_values_scaled = handle_infinity_values(t_values_scaled)  # Handle infinite values
    t_values_scaled = custom_normalize(t_values_scaled)  # Normalize to -1 to +1

    # print(t_values_scaled)
    # quit()
    # T map
    header, maps = bvbabel.create_cmp(nr_maps=1)
    maps["map1"].update({
        "ContrastColor": "#ff0000",
        "MapName": f"{base_name}_tvalues",
        "DF1": 134,
        "DF2": 134,
        "NrOfChannels": 134,
        "NrOfDetectors": 60,
        "NrOfSources": 32,
        "NrOfWaveLength": 3,
        "SDKey": processed_sdkey,
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
        "wl0_cmp": t_values,
        "wl1_cmp": [0]*len(t_values),
        "wl2_cmp": [0]*len(t_values)
    })

    # Combine header and maps and write to file
    out_fname = os.path.join(input_folder, f"{base_name}_tvalues.cmp")
    
    bvbabel.write_cmp(out_fname, header, maps)
  
    # Cohen's D map
    header, maps = bvbabel.create_cmp(nr_maps=1)
    maps["map1"].update({
        "ContrastColor": "#ff0000",
        "MapName": f"{base_name}_cohens_d",
        "DF1": 134,
        "DF2": 134,
        "NrOfChannels": 134,
        "NrOfDetectors": 60,
        "NrOfSources": 32,
        "NrOfWaveLength": 3,
        "SDKey": processed_sdkey,
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
        "wl0_cmp": cohens_ds,
        "wl1_cmp": [0]*len(cohens_ds),
        "wl2_cmp": [0]*len(cohens_ds)
    })

    # Combine header and maps and write to file
    out_fname = os.path.join(input_folder, f"{base_name}_CohensD.cmp")

    bvbabel.write_cmp(out_fname, header, maps)

    # T map scaled
    header, maps = bvbabel.create_cmp(nr_maps=1)
    maps["map1"].update({
        "ContrastColor": "#ff0000",
        "MapName": f"{base_name}_tvalues_scaled",
        "DF1": 134,
        "DF2": 134,
        "NrOfChannels": 134,
        "NrOfDetectors": 60,
        "NrOfSources": 32,
        "NrOfWaveLength": 3,
        "SDKey": processed_sdkey,
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
        "wl0_cmp": t_values_scaled,
        "wl1_cmp": [0]*len(t_values_scaled),
        "wl2_cmp": [0]*len(t_values_scaled)
    })

    # Combine header and maps and write to file
    out_fname = os.path.join(input_folder, f"{base_name}_tvalues_scaled.cmp")
    
    bvbabel.write_cmp(out_fname, header, maps)
