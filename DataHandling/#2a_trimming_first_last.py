"""
fNIRS Data Trimming Script with Automatic Task Detection and Robust Error Handling

This script processes .snirf files to extract and trim raw fNIRS data based on 
experiment annotations, with task-specific processing for optimal analysis preparation.

FUNCTIONS:
1. resting_state_trim(): 
   - Trims data between first and last annotations
   - Sets all annotation durations to zero (no task events needed)
   - Removes all annotations after cropping (clean resting state data)
   - Output: '_TrimmedToEvents.snirf' suffix

2. motor_action_trim():
   - Finds experiment start ('8') and end ('9') annotations
   - Removes annotations outside experiment window and resting annotations ('1')
   - Renames motor annotations ('2' → 'MotorAction') with 16s duration
   - Trims first 10s for steady state, shifts annotations to compensate for cropping
   - Comprehensive safety checks (9 annotations expected, timing validation)
   - Output: '_TrimmedToEventsStSt.snirf' suffix (indicates steady-state trimming)

INPUT: Folder with task-organized .snirf files
OUTPUT: Task-specific trimmed files maintaining folder structure

ERROR HANDLING:
- OverflowError: Corrupted annotation times → '_ProbOverflow'
- KeyError: File corruption/missing data → '_ProbKeyError' 
- ValueError: Missing/invalid annotations → '_ProbInvTim'/'_ProbNoAnnot'

AUTOMATIC TASK DETECTION:
Script detects task type from folder names and applies appropriate processing method.
Maintains organized output structure with descriptive suffixes.

NOTE: Extensively tested and validated with Satori. Includes annotation shift correction for MNE cropping behavior and comprehensive safety validation.

!NEW ADDITION! 26/Jun/2025
*Tested and Validated with Satori/Manual*
- Preserves auxiliary data from input SNIRF files, trims based on specified time range, and writes to output SNIRF files.
- It was added becuase mne-nirs write_raw_snirf does not preserve auxiliary data.
- Aux data are really important for fNIRS data analysis. so this its crucial that they are presserved and trimmed correctly.
"""
import numpy as np
import mne
from mne_nirs.io.snirf import write_raw_snirf
import os
import sys
from PySide6.QtWidgets import QApplication, QFileDialog
import sys
import shutil
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

def resting_state_trim(input_folder, output_folder):
    """   
    Trims resting state fNIRS data between first and last annotations.
    Sets annotation durations to zero and removes all annotations after cropping.
    Handles problematic files by copying with descriptive error suffixes.
    Output: Clean resting state data with '_TrimmedToEvents.snirf' suffix.
    """
    
    print("Resting state trimming started.")
    for file in os.listdir(input_folder):
        if file.endswith(".snirf"):
            input_file = os.path.join(input_folder, file)
            print(f"Processing file: {input_file}")

            try:
                # Read the raw snirf file
                raw = mne.io.read_raw_snirf(input_file, preload=True)
                print(f"Successfully loaded {file}")
                if len(raw.annotations) > 0: 
                    print(f"   Found {len(raw.annotations)} annotations")
                    # Get the onset time of the first annotation
                    first_onset = raw.annotations.onset[0]
                    last_onset = raw.annotations.onset[-1]
                    print(f"    Original duration: {raw.times[-1]:.2f}s")
                    print(f"    First annotation: {first_onset:.2f}s")
                    print(f"    Last annotation: {last_onset:.2f}s")
                    print(f"    Final duration should be {last_onset - first_onset:.4f}s")                
                    # Safety check
                    if first_onset >= last_onset:
                        raise ValueError(f"Invalid annotation times: first ({first_onset}) >= last ({last_onset})")

                    
                    # Set all annotation durations to zero; The resting state has no task, so they are not needed
                    raw.annotations.duration[:] = 0
                    print("     After setting durations to zero:")
                    for annotation in raw.annotations:
                        print(f"   Annotation: {annotation['description']} at {annotation['onset']:.2f}s with duration {annotation['duration']:.2f}s")
                    # Example of annotations:
                    # Annotation: 1 at 10.79s with duration 0.00s
                    # Annotation: 1 at 10.79s with duration 0.00s
                    # Annotation: 9 at 310.77s with duration 0.00s

                    # Crop the Raw object to start at the first annotation onset
                    raw.crop(tmin=first_onset, tmax=last_onset)
                    print("     After cropping:")
                    for annotation in raw.annotations:
                        print(f"   Annotation: {annotation['description']} at {annotation['onset']:.2f}s with duration {annotation['duration']:.2f}s")


                    # Delete the rest of the annotations that remain in the raw object
                    raw.annotations.delete(range(len(raw.annotations)))
                    print(raw.times[0], raw.times[-1])
                    print(f"   Final duration: {raw.times[-1]:.4f}s")
                    # Save the processed file
                    output_file = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_TrimmedToEvents.snirf")
                    write_raw_snirf(raw, output_file)
                    print(f"Processed and saved: {output_file}")

                    # Now we want to preserve the auxiliary data from the input file, so we open the input file copy the stuff we want, change what must be changes (or trimmed) then write in the output the auxiliary
                    preserve_aux_data_and_trim(input_file, output_file, first_onset, last_onset)

                else:
                    raise ValueError(f"No annotations found in file {file}. Skipping this file.")
            except OverflowError as e:
                # This is a catch for the OverflowError that can occur when the file is too large
                # and the annotations cannot be processed correctly
                # This is a pecuilar case of mne library, so we handle it separately through satori
                print(f"OverflowError for file {file}: {e}")
                suffix = "_ProbOverflow"
                problematic_output = os.path.join(output_folder, f"{os.path.splitext(file)[0]}{suffix}.snirf")
                shutil.copy2(input_file, problematic_output)
                continue
            
            except ValueError as e:
                # there is onesubject with missing first annotation so we handle it manually
                error_message = str(e)
                if "Invalid annotation times" in error_message:
                    print(f"ValueError for file {file}: {e}")
                    suffix = "_ProbInvTim"
                elif "No annotations found" in error_message:
                    print(f"ValueError for file {file}: {e}")
                    suffix = "_ProbNoAnnot"
                problematic_output = os.path.join(output_folder, f"{os.path.splitext(file)[0]}{suffix}.snirf")
                shutil.copy2(input_file, problematic_output)
                continue

def motor_action_trim(input_folder, output_folder):
    """ 
    In motor task the experiment:
        - onset has the name: 8; experiment end has the name:9
        - resting task has the name: 1; motor task has the name: 2
            - start: 8; end: 9; resting: 1; motor: 2
    This script deletes the start and the end annotation, which are not needed for the analysis.
    Also it deletes any annotations before the first onster: 8. 
    Crops the data to the first and last annotation onsets, which are 8 and 9.
    Then it shifts the annotation onsets, because after cropping, data starts at 0s, but the annotations still have their original onsets and mne doesnt compensate for that automatically.
    New Addition:
        - After cropping the data and shifting the annotation onsets
        - Delete the rest annotation
        - Set duration of motor annotation to 16s as in experiment protocol
        - Set Name of motor annotation to 'MotorAction' for consistency
        - Trim first 10 seconds of the data, to reach steady state of the motor action; This is valid for the specific paradigm, because there is a fixed 20s duration at the beginning before the first motor action task.
        --> These steps are done to increase consistency of the data, and avoid extra processing steps in later stages (i.e., durations, naming, debugging problematic files that are not processed here)
    Note: the code is tested and validated with Satori; the trimming output is makes logical sense, first motor action occurs at 20s, in the experimental protocol it also starts at 20s.
    Note: The annotation durations are triple checked to be correct. IMPORTANT when checking: resting between motor action has a varied steplength 15-17 seconds, so the total timecourse per subject will be a bit different, but the motor action annotations are correctly set to occur when they occur and last 16 seconds
    Note: the code includes debuging print statements for other's to check the correctness of the trimming and annotation shifting.
    Note: the code includes error handling and error messages to identify problems. When (if) things get catastrophic to avoid missing the problem the code quits entirely 
    Output:
        - trimmed .snirf files with suffix '_TrimmedToEventsStSt' to indicate that the data is trimmed to first and last annotations, and the first 10s are trimmed to reach steady state
        - if the file is problematic, it is copied to the output folder with a suffix indicating the problem (e.g., '_ProbOverflow', '_ProbKeyError', '_ProbInvTim', '_ProbNoAnnot')
        - trimmed files include the timecourse between start and end annotations, 
        with the first 10 seconds trimmed to reach steady state, and the motor action annotation 
        set to 'MotorAction' with a duration of 16 seconds.

    """
    # Example of annotations:
    #    Annotation: 1 at 3.17s with duration 10.00s
    #    Annotation: 8 at 3.17s with duration 10.00s
    #    Annotation: 2 at 23.17s with duration 10.00s
    #    ...
    #    Annotation: 1 at 303.23s with duration 10.00s
    #    Annotation: 9 at 319.19s with duration 7.31s
    print("Motor action trimming is not implemented yet.")
    for file in os.listdir(input_folder):
        if file.endswith(".snirf"):
            input_file = os.path.join(input_folder, file)
            print(f"Processing file: {input_file}")

            try:
                # Read the raw snirf file
                raw = mne.io.read_raw_snirf(input_file, preload=True)
                print(f"Successfully loaded {file}")
                if len(raw.annotations) > 0: # not really needed, but just to be sure; 
                    print(f"   Found {len(raw.annotations)} annotations")
                    # Get the onset time of the first annotation
                    for annotation in raw.annotations:
                        if annotation['description'] == '8':
                            first_onset = annotation['onset']
                        elif annotation['description'] == '9':
                            last_onset = annotation['onset']
                    print(f"    First annotation: {first_onset:.2f}s")
                    print(f"    Last annotation: {last_onset:.2f}s")
                    print(f"    Original duration: {raw.times[-1]:.2f}s")
                    print(f"    Final duration should be {last_onset - first_onset - 10:.4f}s") # 10 are trimmed from the beginning to reach steady state
                    final_duration_check = last_onset - first_onset - 10
                    # Safety check
                    if first_onset >= last_onset:
                        raise ValueError(f"Invalid annotation times: first ({first_onset}) >= last ({last_onset})")
                    # print(f"Before processing annotations {raw.times[-1]:.2f}:")
                    # for annotation in raw.annotations:
                    #     print(f"   Annotation: {annotation['description']} at {annotation['onset']:.2f}s with duration {annotation['duration']:.2f}s")
                    
                    # Collect indices of annotations to delete; this is a more robust way to handle annotations 
                    # otherwise we introduce bugs [e.g., the indices of the annotations change after deleting, 
                    # so if we delete the first instantly the idx is reduced by 1; if we start at the end the idx from 
                    # the begining remain as is and doesnt complicate the deletion]
                    indices_to_delete = []
                    for idx, annotation in enumerate(raw.annotations):
                        if annotation['onset'] <= first_onset or annotation['onset'] >= last_onset:
                            indices_to_delete.append(idx)
                    # basically we delete the first and the last annotation with the <= or >= condition and we are ready to crop based on their onsets
                    # Delete in reverse order to avoid index shifting!!!
                    for idx in reversed(indices_to_delete):
                        raw.annotations.delete(idx)
                    
                    # Now we want to delete the rest of the annotations that are not needed: resting state (1) and rename motor action (2) annotation to 'MotorAction' and set its duration to 16s as in the experiment protocol
                    rest_indices = []
                    for idx, annotation in enumerate(raw.annotations):
                        if annotation['description'] == '1':
                            rest_indices.append(idx)
                    for idx in reversed(rest_indices):
                        raw.annotations.delete(idx)

                    raw.annotations.rename({'2': 'MotorAction'})
                    raw.annotations.set_durations(16)
                    # print(f"After deleting/renaming annotations with time: {raw.times[-1]:.2f}")
                    # for annotation in raw.annotations:
                    #     print(f"   Annotation: {annotation['description']} at {annotation['onset']:.2f}s with duration {annotation['duration']:.2f}s")                    
                    
                    # Crop the raw object to the start + flat 10s to reach steady state till the end of the experiment
                    # The shift all annotations as needed
                    # We do this in one step to (i) avoid multiple cropping and shifting (ii) to ensure the mne library handles it correctly
                    raw.crop(tmin=first_onset+10, tmax=last_onset)
                    raw.annotations.onset -= (first_onset + 10)  # Shift all annotations to start at 10s after cropping
                    
                    # print(f"After cropping, time: {raw.times[-1]:.2f}s; len of ann = {len(raw.annotations)}")
                    # for annotation in raw.annotations:
                    #     print(f"   Annotation: {annotation['description']} at {annotation['onset']:.2f}s with duration {annotation['duration']:.2f}s")

                    # Final Safety checks
                    if len(raw.annotations) != 9:
                        print(f"Warning: Expected 9 annotations after processing, but found {len(raw.annotations)}. Check the trimming logic.")
                        quit()
                    if not np.isclose(raw.annotations.onset[0], 10.0, atol= 0.1): # floating point arithimetic is iherently imprecise so we check if its close to 10s with a tolerance of 0.1s
                        print(f"Warning: First annotation onset is not at 10s, but at {raw.annotations.onset[0]:.10f}s. Check the trimming logic.")
                        quit()
                    if not np.isclose(raw.times[-1], final_duration_check, atol= 0.1):
                        print(f"Warning: Last annotation onset is not at {final_duration_check:.10f}s, but at {raw.times[-1]:.10f}s. Check the trimming logic.")
                        quit()

                    # # Save the processed file
                    output_file = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_TrimmedToEventsStSt.snirf")
                    write_raw_snirf(raw, output_file)
                    print(f"Processed and saved: {output_file}")

                    # Now we want to preserve the auxiliary data from the input file, so we open the input file copy the stuff we want, change what must be changes (or trimmed) then write in the output the auxiliary
                    preserve_aux_data_and_trim(input_file, output_file, first_onset + 10, last_onset)

                else:
                    raise ValueError(f"No annotations found in file {file}. Skipping this file.")
            except OverflowError as e:
            #     # This is a catch for the OverflowError that can occur when the annotation onset is corrupted
            #     # and the annotations cannot be processed correctly
            #     # This is a pecuilar case of mne library, so we handle it separately through satori
                print(f"OverflowError for file {file}: {e}")
                suffix = "_ProbOverflow"
                problematic_output = os.path.join(output_folder, f"{os.path.splitext(file)[0]}{suffix}.snirf")
                shutil.copy2(input_file, problematic_output)
                continue
            except KeyError as e:
                # This is a catch for the KeyError that can occur when the file corrupted somehow and mne cannot read it
                # there is one subject with corrupted file
                print(f"KeyError for file {file}: {e}")
                suffix = "_ProbKeyError"
                problematic_output = os.path.join(output_folder, f"{os.path.splitext(file)[0]}{suffix}.snirf")
                shutil.copy2(input_file, problematic_output)
                continue
            
            except ValueError as e:
                error_message = str(e)
                if "Invalid annotation times" in error_message:
                    print(f"ValueError for file {file}: {e}")
                    suffix = "_ProbInvTim"
                elif "No annotations found" in error_message:
                    print(f"ValueError for file {file}: {e}")
                    suffix = "_ProbNoAnnot"
                problematic_output = os.path.join(output_folder, f"{os.path.splitext(file)[0]}{suffix}.snirf")
                shutil.copy2(input_file, problematic_output)
                continue

# Select the folder that contains the Reorganized data with structure: Reorganized -> TaskType - > SubjectID
input_folder = r"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset\1_Data_Actual_Reorganized"
# create an output folder in the same directory as the input folder
output_folder = os.path.join(os.path.dirname(input_folder), f"{os.path.basename(input_folder)}_TrimmedToEvents")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created output folder: {output_folder}")

for folder in os.listdir(input_folder):
    # check if the folder starts with Resting or Motor
    if "Resting" in folder:
        # if resting state folder, then perform the resting state trimming
        # create a subfolder in the output folder with the same name to keep the structure and save the data there
        resting_subfolder_name = f"{folder}_TrimmedToEvents"
        resting_subfolder_path = os.path.join(output_folder, resting_subfolder_name)
        if not os.path.exists(resting_subfolder_path):
            os.makedirs(resting_subfolder_path)
            print(f"Created subfolder: {resting_subfolder_path}")
        # perform the trimming on the resting state data and save them to the resting_subfolder_path
        resting_input_folder = os.path.join(input_folder, folder)
        resting_state_trim(resting_input_folder, resting_subfolder_path)
    if "MotorAction" in folder:
        # if motor action folder, then perform the motor action trimming
        # create a subfolder in the output folder with the same name to keep the structure and save the data there
        motor_subfolder_name = f"{folder}_TrimmedToEventsStSt" # StSt means Steady State; trimmed first 10s of the data to reach steady state
        motor_subfolder_path = os.path.join(output_folder, motor_subfolder_name)
        if not os.path.exists(motor_subfolder_path):
            os.makedirs(motor_subfolder_path)
            print(f"Created subfolder: {motor_subfolder_path}")
        # perform the trimming on the motor action data and save them to the motor_subfolder_path
        motor_input_folder = os.path.join(input_folder, folder)
        motor_action_trim(motor_input_folder, motor_subfolder_path)
