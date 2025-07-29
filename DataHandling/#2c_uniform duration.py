"""
===============================================================================
Uniform Duration Trimming for fNIRS Resting-State Data
===============================================================================

PURPOSE:
This script ensures all resting-state fNIRS files have identical durations by 
cropping them to the shortest file's duration. This is essential for group-level 
analysis where all subjects must have the same temporal dimensions.

PROCESS OVERVIEW:
1. Load all SNIRF files and identify:
   - Duration of each file
   - Sampling frequency (should be consistent across files)
   - Minimum duration across all subjects

2. CALCULATE EXACT CUTOFF: 
   - Use minimum duration as target length
   - Convert to exact sample number: target_sample = duration * sampling_frequency
   - Convert back to precise time: actual_target_time = target_sample ÷ sampling_frequency
   - This ensures sample-accurate cropping (no rounding errors)

3. For each file:
   - Load OD SNIRF file
   - Crop from 0 to actual_target_time
   - Save as new file with "_UnifDurat" suffix

WHY SAMPLE-ACCURATE CROPPING MATTERS:
- fNIRS data is sampled at discrete time points (e.g., 12.59 Hz)
- Arbitrary time cuts (e.g., exactly 297.0s) may fall between samples
- Sample-accurate cuts ensure all files have identical sample counts
- Critical for matrix operations in PCA/ICA (all subjects must have same dimensions) and othe ranalyses

INPUT:  OD Trimmed SNIRF RS files (varied durations; floating point innaccuracies + problematic files trimmed manually)
OUTPUT: Uniform duration SNIRF files (same duration, ready for group analysis)
===============================================================================
"""
import mne
from mne_nirs.io.snirf import write_raw_snirf
import os
import numpy as np
import h5py

def preserve_aux_data_and_trim(input_file, output_file, start_time, end_time):
    """
    Preserves auxiliary data from the input SNIRF file and trims the raw data to the specified time range.
    
    Parameters:
    - input_file: Path to the input SNIRF file.
    - output_file: Path of saved SNIRF file.
    - start_time: Start time for trimming (in seconds).
    - end_time: End time for trimming (in seconds).
    
    Returns:
    - None
    It adds the auxiliary data in the output file and trims the auxiliary data to the specified time range.
    """
    # SNIRF is already been written by mne-nirs write command, so we can use h5py to add to this already written file the auxiliary from the input.
    print("Adding and trimming auxiliary data...")
    with h5py.File(input_file, 'r') as src, h5py.File(output_file, 'r+') as dst:  # r+ not w
    # Why 'r+' not 'w': Because MNE-NIRS already wrote the main data structure, we just want to add auxiliary data to the existing file, not overwrite it
        src_nirs = src["nirs"]
        dst_nirs = dst["nirs"]
        
        aux_groups = [k for k in src_nirs.keys() if k.startswith('aux')]
        
        aux_copied = 0
        for aux_key in aux_groups:
            try:
                if aux_key in dst_nirs:
                    del dst_nirs[aux_key]
                aux_group = src_nirs[aux_key]
   
                # # Here we will manipulate the auxiliary data that we want to, and leave the others as is.
                # the ones of interest are: Aux1 to Aux6 (motion parameters) & Aux12 (respiratory)
                # dataTimeSeries: shape=(33603, 1), dtype=float64 || these are the actual sensor measurements at each timepoint
                # name: shape=(), dtype=object. must be decoded utf-8
                # time: shape=(33603,), dtype=float64 || these are the timestamps for each auxiliary datapoint
                # # =====================================================
                # aux_time[0] = 0.000 → dataTimeSeries[0] = [1.709] (accelerometer reading at t=0)
                # aux_time[1] = 0.010 → dataTimeSeries[1] = [1.703] (accelerometer reading at t=0.01s)
                # aux_time[2] = 0.020 → dataTimeSeries[2] = [1.686] (accelerometer reading at t=0.02s)
                
                if "aux" in aux_key:
                    # print(f"  Manipulating auxiliary group: {aux_key}")
                    aux_data = aux_group['dataTimeSeries'][()]
                    aux_time = aux_group['time'][()]
                    

                    # Find the index where aux_time is greater than or equal to start_time and end_time
                    # this way we can identify the index in the auxiliary_time that corresponds to the onset and offset of trimming!
                    start_idx = np.where(aux_time >= start_time)[0][0]
                    end_idx = np.where(aux_time >= end_time)[0][0]
                    # # find the index where aux_time is 10 seconds or more
                    # print(f" start_idx: {start_idx}, end_idx: {end_idx}")
                    # print(f"    Trimming auxiliary data from {aux_time[start_idx]:.2f}s to {aux_time[end_idx]:.2f}s")
                    # print(f"    Trimming dataTimeSeries from {aux_data[start_idx]} to {aux_data[end_idx]}")
                    
                    # Having found this index that corresponds to the start/end. We can trim the dataTimeSeries and time arrays, because they have the same indexing
                    trimmed_aux_data = aux_data[start_idx:end_idx+1] # +1 to include the end index
                    trimmed_aux_time = aux_time[start_idx:end_idx+1]  # +1 to include the end index
                    # print(f"    Trimmed dataTimeSeries: {trimmed_aux_data[0]} to {trimmed_aux_data[-1]}")
                    # print(len(trimmed_aux_data), len(trimmed_aux_time))
                    
                    # now we have to "reset" the time to start at - trim_start
                    trimmed_aux_time -= start_time

                    # Create the auxilary group in the destination file
                    new_aux_group = dst_nirs.create_group(aux_key)

                    # Write the trimmed time and data
                    new_aux_group.create_dataset('dataTimeSeries', data=trimmed_aux_data)
                    new_aux_group.create_dataset('time', data=trimmed_aux_time)

                    # Copy other datasets unchanged (name, dataUnit, etc.)
                    for dataset_name in aux_group.keys():
                        if dataset_name not in ['time', 'dataTimeSeries']:
                            # Copy non-time datasets as-is
                            aux_group.copy(dataset_name, new_aux_group)
                    
                    # Copy attributes if any exist
                    for attr_name, attr_value in aux_group.attrs.items():
                        new_aux_group.attrs[attr_name] = attr_value
                              
                    aux_copied += 1
                
                else:
                    # For non-auxiliary groups, copy as-is
                    src_nirs.copy(aux_key, dst_nirs)
                    aux_copied += 1

                print()  # Empty line for readability
                
            except Exception as e:
                print(f"Failed to copy {aux_key}: {e}")
                # Don't quit on individual failures, continue with other groups

        print(f"    Successfully processed {aux_copied} auxiliary groups")

    print("Done!")


# the input folder of my RS data; customize as needed
input_folder = r"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset\1_Data_Actual_Reorganized_TrimmedToEvents\01_RestingState_TrimmedToEvents"

output_folder = os.path.join(os.path.dirname(input_folder), f"{os.path.basename(input_folder)}_UniDu")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created output folder: {output_folder}")

durations = []
samp_freq = []
for file in os.listdir(input_folder):
    if file.endswith(".snirf"):
        input_file = os.path.join(input_folder, file)
        raw = mne.io.read_raw_snirf(input_file, preload=True)
        print(f"Successfully loaded {file}")
        print(f"    duration: {raw.times[-1]:.2f}s")
        durations.append(raw.times[-1])
        sfreq =raw.info['sfreq']
        samp_freq.append(sfreq)

print(f"Total files processed: {len(durations)}")
print(f"min duration: {min(durations):.2f}s")
print(f"max duration: {max(durations):.2f}s")
for i in range(len(durations)):
    print(f"File {i+1}: {durations[i]:.7f}s")
    print(f"    Sample frequency: {samp_freq[i]:.2f}Hz")


cutoff = int(min(durations)) # we use the minumum duration as the cutoff because it ensures uniform duration across files; e.g., if one file is 300s and another is 297s, we will cut both to 297s (if we cut to 300s, the 297s will not be uniform)
# Safety check: sample frequency should be consistent across all files
for freq in samp_freq:
    if freq != samp_freq[0]:
        print(f"Sample frequencies are inconsistent: {freq} != {samp_freq[0]}")
        print("Exiting script to avoid issues with inconsistent sample frequencies...")
        quit()

target_sample = int(cutoff * samp_freq[0])  # After ensuring all files have the same sample frequency
actual_target_time = target_sample / samp_freq[0]
print(f"Taking in account sampling frequency, snirf files will be cut to: {actual_target_time}")
# cut every file to a uniform duration
new_durations = []
for file in os.listdir(input_folder):
    if file.endswith(".snirf"):
        input_file = os.path.join(input_folder, file)
        raw = mne.io.read_raw_snirf(input_file, preload=True)
        raw.crop(tmin=0, tmax=actual_target_time)
        print(f"    Cropped duration: {raw.times[-1]:.2f}s")
        new_durations.append(raw.times[-1])
        # We have to clean the name of the file to avoid long names and writting permission from windows
        pos_clean_name = file.find("_Trim") # this is not dynamic, it only works for RestingState within filename
        clean_name = file[:pos_clean_name] + "_UniDu"
        
        output_file = os.path.join(output_folder, f"{clean_name}.snirf")
        write_raw_snirf(raw, output_file)
        print(f"Processed and saved: {output_file}")
        preserve_aux_data_and_trim(input_file, output_file, 0, actual_target_time)
print(f"if target duration is {actual_target_time}s, then the new durations are:")

for i in range(len(new_durations)):
    print(f"File {i+1}: {new_durations[i]}s")
    if new_durations[i] != actual_target_time:
        raise ValueError(f"File {i+1} has a different duration: {new_durations[i]}s != {actual_target_time}s")
    