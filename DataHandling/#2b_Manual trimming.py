"""
Here i will manually trim the data to the first and last annotation

The purpose of this script is to handle the case where '#1a_trimming_first_last.py' threw an error
and the data was not trimmed.

To be able to handle them first they were placed in Satori for manual inspection
These problematic files processed in Satori
RestingState:
    - S04_01_RestingState_ProbOverflow_RemovedOverflow
    - S13_01_RestingState_ProbOverflow_RemovedOverflow
    --> Overflown annotations were removed; Converted to OD.
Motortask:
    - S05_02_MotorAction_ProbOverflow_RemovedOverflow
    - S08_02_MotorAction_ProbKeyError_toOD
    --> Data were converted to OD, also extra annotations in the specific file were already removed. this file was processed MANUALLY IN SATORI

In this script we will process these two files to trim them to the last annotation

Important: mne_nirs write supports only OD data 
    NOT RAW NOT CC data
"""

import mne
from mne_nirs.io.snirf import write_raw_snirf
import os
import sys
import h5py
import numpy as np

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



# # Resting State files: s04; s13 [Overflow removed and converted to OD]
# # input_file = r"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset\1_Data_Actual_Reorganized_TrimmedToEvents\01_RestingState_TrimmedToEvents\S13_01_RestingState_ProbOverflow_OK_OD.snirf"
# input_file=r"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset\1_Data_Actual_Reorganized_TrimmedToEvents\01_RestingState_TrimmedToEvents\S04_01_RestingState_ProbOverflow_OK_OD.snirf"
# raw = mne.io.read_raw_snirf(input_file, preload=True)
# print("Annotations found:")
# for annotation in raw.annotations:
#     print(f"   Annotation: {annotation['description']} at {annotation['onset']:.2f}s with duration {annotation['duration']:.2f}s")
# raw.annotations.duration[:] = 0
# onset = raw.annotations.onset[0]
# print(f" after trimming it should have: {raw.times[-1] - (raw.times[-1] - onset):.2f}s")
# raw.crop(tmin=0, tmax=onset)
# print(f"after cropping duration: {raw.times[-1]:.2f}s")
# raw.annotations.delete(range(len(raw.annotations)))  # delete all annotations
# print("check annotations:")
# for annotation in raw.annotations:
#     print(f"   Annotation: {annotation['description']} at {annotation['onset']:.2f}s with duration {annotation['duration']:.2f}s and total duration: {raw.times[-1]:.2f}s")
# # Save the trimmed data with the new filename
# output_file = input_file.replace(".snirf", "_TrimToEvents.snirf")
# write_raw_snirf(raw, output_file)
# print(f"Saved trimmed data to {output_file}")
# # Preserve auxiliary data and trim the raw data
# preserve_aux_data_and_trim(input_file, output_file, start_time=0, end_time=raw.times[-1])

# # motor task files: s05 [Overflow removed and converted to OD]. This specific file is missing the start_trigger 8; the motor task event (2) starts at 18.25s; the file is trimmed to preserve timepoints: 8.25 to end_trigger (9) to account for SteadyState and having uniform motor task event starting point with the rest of the data
# input_file = r"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset\1_Data_Actual_Reorganized_TrimmedToEvents\02_MotorAction_TrimmedToEventsStSt\S05_02_MotorAction_ProbOverflow_OK_OD.snirf"
# raw = mne.io.read_raw_snirf(input_file, preload=True)
# print("Annotations found:")
# for annotation in raw.annotations:
#     print(f"   Annotation: {annotation['description']} at {annotation['onset']:.2f}s with duration {annotation['duration']:.2f}s")
#     if annotation['description'] == '9':
#         last_onset = annotation['onset']
# print(f"The duration before cropping: {raw.times[-1]:.2f}s")
# print(f"The duration after cropping should be: {raw.times[-1] - 8.25 - (raw.times[-1] - last_onset):.2f}s")
# indices_to_delete = []
# for idx, annotation in enumerate(raw.annotations):
#     if annotation['description'] in ['1', '9']:
#         indices_to_delete.append(idx)
# print(f"Deleting annotations with indices: {indices_to_delete}")
# for idx in reversed(indices_to_delete):
#     raw.annotations.delete(idx)

# raw.annotations.rename({'2': 'MotorAction'})
# raw.annotations.set_durations(16)

# raw.crop(tmin=8.25, tmax=last_onset)  # Crop to the first motor action start and last annotation
# raw.annotations.onset -= 8.25  # Adjust annotations to start at 0s
# for annotation in raw.annotations:
#     print(f"   Annotation: {annotation['description']} at {annotation['onset']:.2f}s with duration {annotation['duration']:.2f}s and total duration: {raw.times[-1]:.2f}s")

# output_file = input_file.replace(".snirf", "_TrimToEventsStSt.snirf")
# write_raw_snirf(raw, output_file)
# print(f"Saved trimmed data to {output_file}")

## motor task files: s08, s30 [Converted to OD]
# s08
# input_file = r"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset\1_Data_Actual_Reorganized_TrimmedToEvents\02_MotorAction_TrimmedToEventsStSt\S08_02_MotorAction_ProbKeyError_OD.snirf"
# raw = mne.io.read_raw_snirf(input_file, preload=True)
# print(f"time before cropping: {raw.times[-1]:.2f}s")
# print(f"time after cropping should be {351.932165 - (31.822081 + 10):.2f}s")
# print("Annotations found:")
# raw.annotations.set_durations(16)
# raw.annotations.rename({'MotorPerformance': 'MotorAction'})
# #         # Start = MotorOnset - BaselineFixation = 51.822081 - 20 = 31.822081
# #         # End = MotorOffset + BaselineFixation = 315.932165 + 16 + 20 = 351.932165
# raw.crop(tmin=31.822081 + 10, tmax=351.932165)  # Crop to the first motor action start and last annotation and trim first 10s to reach stable state
# raw.annotations.onset -= (31.822081 + 10)  # Adjust annotations to start at 0s
# for annotation in raw.annotations:
#     print(f"   Annotation: {annotation['description']} at {annotation['onset']:.2f}s with duration {annotation['duration']:.2f}s")
# print(f"Duration after cropping: {raw.times[-1]:.2f}s")
# output_file = input_file.replace(".snirf", "_TrimToEventsStSt.snirf")
# write_raw_snirf(raw, output_file)
# print(f"Saved trimmed data to {output_file}")

