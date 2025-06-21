import os
import pandas as pd
import shutil
"""
Data Exclusion Script for fNIRS Analysis

This script removes data files for participants who should be excluded from analysis
based on data quality assessment. It reads an Excel file containing inclusion/exclusion 
decisions and deletes corresponding .snirf files from the reorganized data folder.
"""
# directory of the excel with inclusion status and the data quality assessmnet
inclusion_filepath = r"c:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset\1a_perTask_SCI\Quality Assessment.xlsx"
inclusion_df = pd.read_excel(inclusion_filepath, sheet_name="Final Dataset")
inclusion_df = inclusion_df.iloc[:, :2] # Keep only the first two columns Col1: Subject Col2: Inclusion Status
print(inclusion_df.head())
included_participants = inclusion_df['Subject'].tolist()
print(included_participants)


# Mannually create a copy of the actual data folder in this instance i renamed it to "2_Data_AfterExclusion" and process the data in this folder by removing files based on subject name
data_folder = r"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset\2_Data_AfterExclusion"
for task_folder in os.listdir(data_folder): # for task_folder; i.e., RestingState.
    task_folder_path= os.path.join(data_folder, task_folder)
    print(f"Processing task folder: {task_folder_path}")
    for subject_file in os.listdir(task_folder_path):
        subject_name = subject_file.split('_')[0] # Assuming the subject name is the first part of the filename before an underscore which it is
        print(f"Processing subject file: {subject_file} for subject: {subject_name}")
        if subject_name not in included_participants:
            print(f"Excluding subject: {subject_name}")
            subject_file_path = os.path.join(task_folder_path, subject_file)
            os.remove(subject_file_path)
            print(f"Removed file: {subject_file_path}")


# Please verify the data folder after running this script to ensure that the files for excluded subjects have been removed.