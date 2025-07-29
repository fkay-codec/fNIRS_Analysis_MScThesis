"""
This script performs a one-sample t-test to determine whether a specific fNIRS channel is  
consistently explained by a predefined seed-channel across multiple participants.  

### Background:  
- Functional connectivity is assessed using a seed-based approach, where a known seed channel
  (identified by localizer task) is used to predict the time course of other channels.  
- A General Linear Model (GLM) is applied to estimate beta values, which quantify the strength
  of the relationship between the seed and each target channel.  
- Each participant contributes one beta value per channel, reflecting how well the seed explains  
  that channel's activity.
### Statistical Test:  
- A one-sample t-test is performed on the beta values of a specific channel across participants.  
- The null hypothesis (H₀) states that the seed does not significantly predict the channel's time course,  
  meaning the beta values are not different from zero (β = 0).  
- The alternative hypothesis (H₁) suggests that the seed consistently predicts the channel's time course (β ≠ 0).  

### Interpretation:  
- After FDR correction
- If the p-value < 0.05, the seed significantly explains the channel's activity across participants,  
  indicating functional connectivity.  
- If the p-value ≥ 0.05, no significant connectivity is detected.  
"""


import sys
import scipy.stats as stats
import pandas as pd
import os
from statsmodels.stats.multitest import fdrcorrection

# Add the directory containing select_folder.py to the module search path
sys.path.append(r"c:\Users\foivo\Documents\Python\Scripts")

import select_folder as sf

# Set pandas to display floats with higher precision
pd.options.display.float_format = '{:.12f}'.format


input_folder = sf.select_folder_path("Select folder: betas of all subjects in one excel")
if not input_folder:
  print("No folder selected. Exiting.\n")
  sys.exit()
# The output path, will be the same as the input folder path with the grouped betas

# Iterate through each file in the input folder
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

  # Read the Excel file, in the format: channel_name, P01_betas, ..., PXX_betas
  data = pd.read_excel(file_path)
  print(data.head())

  # Initialize lists to store results
  channel_names = []
  t_values = []
  p_values = []
  adjusted_p_values = []
  std_devs = []
  mean_betas = []
  cohen_ds = []

  # Iterate through each row in the DataFrame
  for index, row in data.iterrows():
      # Extract the channel name (first column)
      channel_name = row.iloc[0]
 
      # Extract the beta values (skip the first column)
      beta_values = pd.to_numeric(row[1:], errors='coerce')  # Convert to numeric, coercing errors to NaN
      
      # Check for NaN values and skip the row if necessary
      if beta_values.isna().any():
          print(f"Warning: NaN values detected in channel {channel_name}. Skipping this row.")
          continue
      
      # Calculate the standard deviation of beta values and mean
      std_dev = stats.tstd(beta_values)
      mean_beta = beta_values.mean() # double checked if its correct

      # Perform one-sample t-test (null hypothesis: beta = 0)
      t_stat, p_val = stats.ttest_1samp(beta_values, 0)

      # Compute Cohens'd
      cohen_d = (mean_beta - 0) / std_dev # (mean beta - theoretical mean (0) ) / std deviation
      
      # Store the results
      channel_names.append(channel_name)
      t_values.append(t_stat)
      p_values.append(p_val)
      std_devs.append(std_dev)
      mean_betas.append(mean_beta)     
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
      "Mean Beta": mean_betas,
      "Cohen's d": cohen_ds      
  })

  # Export the DataFrame to an Excel file
  results_df.to_excel(output_file, index=False, header=True)
  print(f"Results exported to {output_file}")

