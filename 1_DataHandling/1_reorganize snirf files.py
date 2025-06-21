import os, shutil

# This script copies all the .snirf files from subject folders across all participants into another task folder while keeping the subject number
# its nice to group the files together for preprocessing and later for analysis

# This script goes through the MULPA dataset, and from each subject takes the .snirf task .snirf file and transfers it in another folder with the task name
# for example, from s01 s02 s03 s04 the .snirf files for motor action are taken and placed in a motor action file and each .snirf is organized by subject number

# change source_dir based on your directory of the raw data from the MULPA project
source_dir = r"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset\1_Data_Actual"
output_dir = os.path.dirname(source_dir)
print("source directory is", output_dir)

destination_dir = os.path.join(output_dir, os.path.basename(source_dir) + "_Reorganized")
if not os.path.exists(destination_dir):
    print("destination directory does not exist, creating it")
    os.makedirs(destination_dir)

# create a list with all the subfolders in the source path that start with 'S' or 'P'
# the subfolders are the subject folders, e.g. S01, S02, P01, P02
subfolders = []
for f in os.listdir(source_dir):
    full_path = os.path.join(source_dir, f)
    if os.path.isdir(full_path):  # Check if it's a folder
        subfolders.append(f)
print("these are the subfolders \n", subfolders)

# this is hardcoded for the MULPA dataset; given the fact that we use a template structure of folders
tasks = ["01_RestingState", "02_MotorAction", "03_MotorImagery", "04_MotorImagery_FreqChange",
         "05_EmotionTask","06_VisualTask", "07_MusicTask1", "08_MusicTask2"]
# task = input("which task folder? USE F2\n")

for task in tasks:
    dest_task_folder_path = os.path.join(destination_dir, task)
    if not os.path.exists(dest_task_folder_path):
        print(f"Creating task folder: {dest_task_folder_path}")
        os.makedirs(dest_task_folder_path)
    for folder in subfolders: # itterate through the subject folders
        # link the source directory with the subjects folder into a completed director
        # access the subjects folder
        subject_folder = os.path.join(source_dir, folder)

        # access the task folder
        task_folder = os.path.join(subject_folder, task)

        if os.path.isdir(task_folder):  # Check if task folder exists; with this line we dont need to worry if a task was not performed.
            for file_name in os.listdir(task_folder): #iterate through the files in the task folder
                if file_name.lower().endswith('.snirf'): # Name of the file to copy that ends with .snirf
                    source_file = os.path.join(task_folder, file_name) # link into a path the motor_action directory with the file name
                    formatted_folder_name = folder.capitalize()  # Ensure consistent naming like S01, S02, etc.
                    destination_file = os.path.join(dest_task_folder_path, f"{formatted_folder_name}_{task}.snirf")
                    shutil.copy2(source_file, destination_file)
                    print(f"Copied {file_name} from {task_folder} to {dest_task_folder_path}")
