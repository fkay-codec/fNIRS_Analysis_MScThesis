# Information about each folder/file

Folder/files are not renamed to make scripts easier to understand. Some scripts used input_folder/input_files and produced output_folders/output_files and the names/structure are preserved in this folder [without the whole directory path]

Folder:
`02a_test_MotorAction_GroupGLM_correctSerialC`
    - contains the group glm results from the motor task

`4_SBA_GLM`
    - contains the results from SBA-GLM
    RAW folders: `SBA_GLM_Deoxy` & `SBA_GLM_Oxy`
        - only excel files
        - Deoxy for S25D23
        - Oxy for S10D7
    `_Grouped_Betas` suffix
        - betas grouped and preped for t-test
        - ttest results

Similar logic for folders `5_SBA_GLM_Resp` & `6_SBA_Corr`

`7_ICA`
    - contains the results from ICA
    `1_FastICA_SM_Seed2025_Runs50`
        - folders of each participant with results from `_logcosh` and `_skew` function
        e.g., folder `P01_01_RestingState_logcosh`
            - contains the results in excel format for each run (initial seed = 2025; for i = 1 to i = 50)
    `6_AllRunsBestICs_PerSubject`
        - folder that contains the best IC per run for each participant
        (participant 1 =  an excel with 50 columns, each column was the best IC of the respective run)
    `7b_IC_Median_AcrossRuns`
        - folder that contains the excels of the median of these 50 runs per participant
`8_PerformanceEvaluation`
    - a copy of the results that where used to compute ROC and Spatial similarity
    - results in text format