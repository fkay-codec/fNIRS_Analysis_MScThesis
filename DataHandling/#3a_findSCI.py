import os
import pandas as pd

"""
This script analyzes Scalp Channel Index (SCI) data from Satori text output.

Main Functions:
1. File Processing: Reads .txt files containing SCI values for different channels
2. Channel Classification: 
   - Maps channels to source-detector (SD) pairs from an Excel reference file (Satori output)
   - Categorizes channels as "short" (D > 28) or "long" (D ≤ 28) distance based on detector values
   - Applies inclusion/exclusion criteria (SCI >= 0.70 = include, < 0.70 = exclude)

Analysis Output:
- Per-participant statistics: Counts of excluded/included channels by distance type
- Percentage calculations: Exclusion rates for long and short distance channels
- Quality assessment: Identifies participants with >30% exclusion rates
- Excel export: Saves summary data to SCI07_PerParticipant.xlsx
"""

def parse_sci_file_lines(file_path):
    channels = []
    sci_values = []

    with open(file_path, 'r') as file:

        for line in file:
            line = line.strip()
            if 'CH:' in line and 'SCI:' in line:
                # Split by 'SCI:' and extract values
                parts = line.split('SCI:')
                ch_part = parts[0].replace('CH:', '').strip()
                sci_part = parts[1].strip()
                
                channels.append(int(ch_part))
                sci_values.append(float(sci_part))
    
    df = pd.DataFrame({
        'Channel': channels,
        'SCI': sci_values
    })
    
    return df

def extract_d_value(channel):
    """Extract the d value from the channel name.
    Example:
        Input: 'S10-D7'
        Output: '7'
    """
    channel = channel.split('-')[-1]  # Extract the last part after the underscore, which is the: D##
    return int(channel[1:]) # Extract the last part after the underscore, which is the d value


# input folder containint the SCI text files of RestingState_OD data 
input_folder = r"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset\1_Data_Actual_Reorganized_TrimmedToEvents\01a_RestingState_SCI_texts"


# extract an excel that has the Ch to SD mapping to use it later
channel_to_SD_path = r"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset\S01 example\01_RestingState\2024-05-02_003_TRIM_GLM_Results.xlsx"
channel_to_SD_data = pd.read_excel(channel_to_SD_path)
channel_to_SD_data = channel_to_SD_data.iloc[1:, [0,1]].reset_index(drop=True).T.reset_index(drop=True).T
channel_to_SD_data.columns = ['CH', 'SD']
channel_to_SD_data['CH'] = channel_to_SD_data['CH'].replace('CH', '', regex=True).astype(int)
#  Extract D values and create the new column
channel_to_SD_data['D_Value'] = channel_to_SD_data['SD'].apply(extract_d_value)
channel_to_SD_data['Distance'] = channel_to_SD_data['D_Value'].apply(lambda x: 'short' if x > 28 else 'long')
print(channel_to_SD_data.head())

participant_data = pd.DataFrame()
for file in os.listdir(input_folder):

    if file.endswith('.txt'):
        file_path = os.path.join(input_folder, file)
        # print(f"Processing file: {file_path}")
        pos_clean_name = file.find('_{')
        clean_name = file[:pos_clean_name-1].split('.')[0]  # Extract the name before the first dot
        print(f"Clean name extracted: {clean_name}")
        data = parse_sci_file_lines(file_path)
        data['Distance']= data['Channel'].apply(lambda x: channel_to_SD_data.loc[channel_to_SD_data['CH'] == x, 'Distance'].values[0] if x in channel_to_SD_data['CH'].values else 'unknown')
        # # print the number of 'short' and 'long' distances
        # print(data['Distance'].value_counts())
        # quit()
        # based on the x number (SCI) assign inclusion or exclusion
        data['Inclusion'] = data['SCI'].apply(lambda x: 'include' if x >= 0.70 else 'exclude')
        # count the number of short and long distances
        distance_counts = data['Distance'].value_counts()
        # count the number of inclusions and exclusions general
        exclusion_counts = data['Inclusion'].value_counts()
        # print(data.to_string())
        # print(exclusion_counts)
        # print(distance_counts)
        # count the numnber of inclusions per distance

        # number of short and long distances:
        short_count = 32
        long_count = 102
        # count the number of inclusions per distance
        inclusion_by_distance = data.groupby(['Distance', 'Inclusion']).size().unstack(fill_value=0)
        # print("\nInclusion/Exclusion by Distance:")
        print(inclusion_by_distance)

        # Get specific counts
        excluded_long = data[(data['Distance'] == 'long') & (data['Inclusion'] == 'exclude')].shape[0]
        included_long = data[(data['Distance'] == 'long') & (data['Inclusion'] == 'include')].shape[0]
        excluded_short = data[(data['Distance'] == 'short') & (data['Inclusion'] == 'exclude')].shape[0]
        included_short = data[(data['Distance'] == 'short') & (data['Inclusion'] == 'include')].shape[0]
        
        # print(f"\nDetailed counts:")
        # print(f"Excluded long channels: {excluded_long}")
        # print(f"Included long channels: {included_long}")
        # print(f"Excluded short channels: {excluded_short}")
        # print(f"Included short channels: {included_short}")

        percent_excluded_long = (excluded_long / long_count) * 100
        percent_excluded_short = (excluded_short / short_count) * 100
        print(f"\nPercentage of excluded long channels: {percent_excluded_long:.2f}%")
        print(f"Percentage of excluded short channels: {percent_excluded_short:.2f}%")
        new_row = pd.DataFrame({
            'Participant': clean_name,
            'Total Channels': len(data),
            'Excluded Long': excluded_long,
            'Included Long': included_long,
            'Excluded Short': excluded_short,
            'Included Short': included_short,
            'Percent Excluded Long': percent_excluded_long,
            'Percent Excluded Short': percent_excluded_short
        }, index=[0])
        participant_data = pd.concat([participant_data, new_row], ignore_index=True)

# Print how many have excluded % above 30 on both long and short distance 

# Count participants with >30% exclusion for long channels
high_excluded_long = participant_data[participant_data['Percent Excluded Long'] >= 30]
count_high_excluded_long = len(high_excluded_long)

# Count participants with >30% exclusion for short channels  
high_excluded_short = participant_data[participant_data['Percent Excluded Short'] >= 30]
count_high_excluded_short = len(high_excluded_short)

# Count participants with >30% exclusion for BOTH long AND short channels
high_excluded_both = participant_data[
    (participant_data['Percent Excluded Long'] >= 30) |
    (participant_data['Percent Excluded Short'] >= 30)
]
count_high_excluded_both = len(high_excluded_both)

print(f"\nEXCLUSION ANALYSIS (>30%):")
print(f"Participants with >30% long channels excluded: {count_high_excluded_long}/{len(participant_data)}")
print(f"Participants with >30% short channels excluded: {count_high_excluded_short}/{len(participant_data)}")
print(f"Participants with >30% excluded in BOTH long OR short: {count_high_excluded_both}/{len(participant_data)}")

# add a new column with the inclusion status based on the exclusion criteria
participant_data['Inclusion Status'] = participant_data.apply(
    lambda row: 'exclude' if (row['Percent Excluded Long'] >= 30 or row['Percent Excluded Short'] >= 30) else 'include',
    axis=1
)




output_file = os.path.join(os.path.dirname(input_folder), 'SCI07_PerParticipant.xlsx')
# Save the participant data to an Excel file
with pd.ExcelWriter(output_file) as writer:
    participant_data.to_excel(writer, index=False, sheet_name='SCI_PerParticipant')

