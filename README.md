# A Quantitative Comparison of Resting State Functional Connectivity Approaches in fNIRS:
# Complete Data Processing & Analysis Pipeline

Author: Foivos Kotsogiannis
Email: f.kotsogiannisteftsoglou@student.maastrichtuniversity.nl
Last Updated: 1/8/2025

## 1. DataHandling folder

### File Processing Pipeline:

1. **`#1a_reorganize snirf files.py`**
    - Reorganizes .snirf files from subject-based to task-based folder structure
    - **Input:** `Subject/Task/file.snirf` 
    - **Output:** `Task/Subject_Task.snirf`

2. **`#2a_trimming_first_last.py`**
   - Automatically detects task type from folder name and applies appropriate trimming
   - **Resting State:** Trims between first/last annotations, sets all annotation durations to zero, removes all annotations (clean resting state data)
   - **Motor Action:** Finds experiment start ('8') and end ('9') annotations, removes annotations outside experiment window, deletes resting state annotations ('1'), renames motor annotations ('2' → 'MotorAction'), sets motor duration to 16s, trims first 10s for steady state, shifts all annotations to start at t=10s
   - **Safety Validation:** Checks annotation count (9 expected), first annotation timing (10.0s ± 0.1s tolerance), and final duration
   - **Auxiliary Data Preservation (June 26, 2025):**
      - Preserves and trims auxiliary data (motion parameters, respiratory signals) from input SNIRF files with proper temporal alignmnet
      - **Problem:** MNE-NIRS `write_raw_snirf()` does not preserve auxiliary data, which is crucial for fNIRS analysis
      - **Implementation:** Uses h5py to extract auxiliary time series, trims to match main data timeframe, resets timestamps, and writes to output file

   - **Output:** 
      - Resting State: `_TrimmedToEvents.snirf` (with preserved auxiliary data)
      - Motor Action: `_TrimmedToEventsStSt.snirf` (indicates steady-state trimming + auxiliary preservation)

3. **Manual Processing in Satori & `#2b_Manual trimming.py` for Problematic Files**
    - Files with OverflowError (corrupted annotation times) processed manually in Satori
    - Delete overflown annotations, convert to OD format
    - **Example Files:**
        - **RestingState:** S04, S13 - Overflown annotations removed + converted to OD; Processed with `#2b_Manual trimming.py`
        - **MotorTask:** S05, S08 - Missing/corrupted annotations; Manually trimmed based on task protocol; converted to OD
    - **Output:** Manually corrected files ready for further processing

4. **`#2c_uniform_duration.py`** for **RestingState** data
   - Identifies minimum duration across all subjects to prevent data loss
   - Performs sample-accurate cropping using exact sample calculations
   - Validates sampling frequency consistency (12.59 Hz expected)
   - **Critical for group analysis:** Ensures identical temporal dimensions for matrix operations (PCA/ICA) & task relevant SCI computation
   - **Output:** Files with `_UnifDurat.snirf` 

5. **Data Format Standardization in Satori**
   - Convert all remaining RestingState/MotorTask data to OD format for uniform processing.
   - **Output:** Files with `_OD.snirf` 

6. **Manual SCI Analysis in Satori**
   - Load all .snirf files into Satori Workflow `RestingStateSCI.flow`.
   - Perform Scalp Channel Index (SCI) analysis with cutoff 1.00
   - **Output:** Individual .txt files per subject containing SCI values for all channels

7. **`#3a_findSCI.py`**
   - Reads SCI .txt files from all participants
   - Categorizes channels as long/short distance based on detector values (D > 28 = short)
   - Applies inclusion criteria: SCI ≥ 0.70 for channel inclusion
   - Calculates percentage of rejected short/long channels per participant
   - **Output:** Excel file with participant-level quality metrics (SCI07_PerParticipant.xlsx)

8. **Manual Data Quality Assessment in Excel file**
   - Open `SCI07_PerParticipant.xlsx` and copy its contents to the `Quality Assessment BioSignalBy JOAO PEREIRA.xlsx`. In that excel Dr. J. Pereira has assessed the quality of respiratory signal of the participants which is used for exclusion (if low quality). Assess which participants will be included in the final analysis based on: 
      (i) poor fNIRS quality, defined as having more than 30% of either long or short channels with a scalp coupling index (SCI) below 0.70. 
      (ii) missing fNIRS data during either resting state or motor action 
      (iii) poor respiratory signal during resting state or motor action
   ==> In the final analysis 31 participants are included.

9. **`#3b_exclude_reorganize.py`**
   - Reads participant inclusion status from quality assessment Excel file `Quality Assessment BioSignalBy JOAO PEREIRA.xlsx`
   - Manually create copy of data folder and the script .snirf files for excluded participants based on quality assessment
   - **Output:** Clean dataset containing only included participants

**NOTES:**
- **Critical Annotation Issue:** MNE/Satori do not automatically shift annotation timing after cropping. Manual correction implemented to ensure events align with trimmed timeline (verified by GLM comparison)
- Error handling preserves problematic files with descriptive suffixes (_ProbOverflow, _ProbKeyError, _ProbNoAnnot, _ProbInvTim) for manual processing
- All data standardized to OD format for uniform processing pipeline

## 2. Satori Pre-Processing of the fNIRS data

### For Seed Based Analysis

#### Motor Action Data:

**Purpose:** Prepare data for identification of seed for SBA analysis.

**Pre-Processing Steps Applied:**

1. **Steady State Trimming**
   - First 10 seconds have been already trimmed (See ## 1. DataHandling folder, Section 2)

2. **Raw Signal to Optical Density Conversion**
   - Convert raw intensity signals to optical density format
   - Already performed in DataHandling (See ## 1. DataHandling folder, Section 5)
   - Prepares data for Beer-Lambert Law transformation

3. **Channel Quality Control**
   - Rejection of channels with Scalp Coupling Index (SCI) < 0.70
   - Ensures adequate signal quality    
   - Maintains consistency with resting-state quality criteria

4. **Optical Density to Hemoglobin Concentration**
   - Transform optical density to hemoglobin concentration using modified Beer-Lambert Law
   - Converts to physiologically meaningful ΔHbO and ΔHbR signals

5. **Bandpass Filtering (0.015 - 0.5 Hz)**
   - Removes contaminating signal from low-frequency noise, respiration, and heart-rate
   - Applied to isolate task-relevant hemodynamic responses
   - References: Khan et al., 2024; Lu et al., 2010; H. Zhang et al., 2010; Y.-J. Zhang et al., 2010

6. **Motion Artifact Correction**
   - **Spike Removal Parameters:**
      - Iterations: 10
      - Lag: 5 seconds
      - Threshold: 3.5
      - Influence: 0.5
      - with monotonic interpolation
   - **Temporal Derivative Distribution Repair (TDDR)** applied to restore high frequencies
   - Removes motion-induced signal corruption while preserving task-related activation

7. **Short-Channel Regression**
   - GLM-based regression using highest correlated short channel
   - Removes systemic physiological artifacts while preserving neuronal signals

8. **Data Normalization**
   - Z-normalization applied to standardize signal amplitude across participants
   - Enables consistent statistical analysis and group comparisons

**Input:** Trimmed Motor task OD data with 9 MotorAction blocks (16s duration each)
**Output:** Preprocessed data with `_SatPreP.snirf` suffix

**Satori Workflow file:** `MotorActionPreprocessing.flow`

#### RestingState Data:
   Similar to the Motor Action Preprocessing, except (4) (5).

**Purpose:** Prepare data for SBA analysis

**Pre-Processing Steps Applied:**

1. **Raw Signal to Optical Density Conversion**
   - Convert raw intensity signals to optical density format
   - Already performed in DataHandling (See ## 1. DataHandling folder, Section 5)

2. **Channel Quality Control**
   - Rejection of channels with Scalp Coupling Index (SCI) < 0.70
   - Ensures adequate signal quality    

3. **Optical Density to Hemoglobin Concentration**
   - Transform optical density to hemoglobin concentration using modified Beer-Lambert Law
   - Converts to physiologically meaningful ΔHbO and ΔHbR signals

4. **Steady State Trimming**
   - First 20 seconds, last 20s trimmed to reach steady state

5. **Bandpass Filtering (0.01 - 0.1 Hz) + Linear Detrending**
   - Extracts resting state signal
   - Remove additional noise

6. **Motion Artifact Correction**
   - **Spike Removal Parameters:**
      - Iterations: 10
      - Lag: 5 seconds
      - Threshold: 3.5
      - Influence: 0.5
      - with monotonic interpolation
   - **Temporal Derivative Distribution Repair (TDDR)** applied to restore high frequencies
   - Removes motion-induced signal corruption while preserving task-related activation

7. **Short-Channel Regression**
   - GLM-based regression using highest correlated short channel
   - Removes systemic physiological artifacts while preserving neuronal signals

8. **Data Normalization**
   - Z-normalization applied to standardize signal amplitude across participants
   - Enables consistent statistical analysis and group comparisons

**Input:** RestingState OD data of uniform duration with no annotations
**Output:** Preprocessed data with `_SatPreP.snirf` suffix

**Satori Workflow file:** `RestingStateSBA.flow`

### For Independent Component Analysis

#### RestingState Data:
   - Create a copy of `"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset\2_Data_AfterExclusion\01_RestingState_TrimmedToEvents_UniDu_OD"`
   - This folder contains .snirf files that have been processed according to ## 1. DataHandling folder --> ### File Processing Pipeline --> Steps 1 to 4.
   - In short snirf files have uniform duration, with no annotations, and are in OD 
   - These .snirf files are  further preprocessed according to the ICA steps below:

**Pre-Processing Steps Applied in Satori:**

1. **Optical Density to Hemoglobin Concentration**
   - Transform optical density to hemoglobin concentration using modified Beer-Lambert Law

2. **Steady State Trimming**
   - First 20 seconds, last 20s trimmed to reach steady state

3. **Broad Bandpass Filtering (0.01 - 0.2 Hz) + Linear Detrending**
   - Extracts resting state signal
   - Remove additional noise

4. **Data Normalization**
   - Z-normalization applied to standardize signal amplitude across participants

**Input:** Copied RestingState OD data of uniform duration with no annotations
**Output:** Preprocessed data with `_SatPreP.snirf` suffix in folder `RestingState_SatPreP_forICA`

See `ICA_Preprocessing.flow`

`NOTE: removal of second order polynomial is done right before ICA analysis [see ~Section 4. ICA~]`

## 3. Seed-Based Analysis (SBA)

### Motor Task Localizer: Seed Identification

**Purpose:** Identify the most significant motor cortex channel for use as seed in resting-state connectivity analysis

**Method:**
1. **Group-Level GLM Analysis**
   - Load all preprocessed Motor Action .snirf files (`_SatPreP.snirf`) into `Satori`
   - Perform Multi-Study/Group GLM with MotorAction vs. baseline + Correct for serial correlation

2. **Seed Selection Criteria**
   - Highest statistical significance in motor cortex region

**Results:**
- **Output folder:** `02a_MotorAction_GroupGLMResults`
- **Selected seed:** 
   - OXY: S10D7 (p < 3.65e-285; t = 46.3)
   - DEOXY: S25D23 (p < 1.3e-151; t = -46.68)
- **Anatomical location - Based on fOLD**
   - S10D7 = Brodmann 6 or premotor
   - S25D23 = Brodmann 6 or premotor
- **Justification:** Most significant and consistent motor activation across the group

### SBA using GLM (SBA-GLM)

**Purpose:** Identify RSFC using the seed as a predictor for HbO/HbR

*Script Folder:* `SBA GLM`

**Method:**
1. **Individual level RSFC maps**
   - loaded all .snirf resting state files with suffix `_SatPreP` to `Satori Workflow`, and perform GLM with:
      Oxy:
         - Predictor: S10D7 timecouse, Motion Parameters (translation, rotation, z-normalization)
         - See `SBA_GLM_Oxy_S10D7.flow`
      Deoxy:
         - Predictor: S25D23 timecouse, Motion Parameters (translation, rotation, z-normalization)
         - See `SBA_GLM_Deoxy_S25D23.flow`
   **Output Saved:**
      - Oxy: `(...)\4_SBA_GLM\SBA_GLM_Oxy`
      - Deoxy: `(...)\4_SBA_GLM\SBA_GLM_Deoxy`
      - Type: `Excel files` with Individual-RSFC maps (betas/t-values/p-values)
   **Note:** Results where verified with manual GLM from one participant. Extract SDM of channel of interest, add SDM as predictor, compare with the results (important: Different SDM for Oxy/Deoxy; make sure to click Oxy/Deoxy)

2. **Prepare Individual RSFC maps for Group Analysis**
   - Open script folder: `SBA t-test`
   - Execute: `#1 betas of all subjects in one excel.py`
      - From the Invidual RSFC map folder, itterate through the results in excel format, and create a single dataframe with all the subject betas
      - Preparing the data for group analysis (i.e., t-test)
   **Output Saved:**
      - Single Excel file: `Group_GLM_Betas_{seed_name}_{chromophore}.xlsx`
      - Structure: Rows = brain channel pairs, Columns = individual participants

3. **Perform Group Analysis using one sample t-test**
   - Open script folder: `SBA t-test`
   - Execute: `#2 SBA t-test.py`
   - **Purpose:** Determine whether specific fNIRS channels' timecourse  are consistently explained by the predefined seed-channel across multiple participants
   - **Statistical Method:**
      - One-sample t-test performed on beta values for each channel across participants
      - Null hypothesis (H₀): Seed does not significantly predict channel's time course (β = 0)
      - Alternative hypothesis (H₁): Seed consistently predicts channel's time course (β ≠ 0)
   - **Multiple Comparison Correction:**
      - False Discovery Rate (FDR) correction using Benjamini-Hochberg method
      - Controls expected proportion of false discoveries among significant findings
      - Addresses multiple testing across 102+ brain channels simultaneously
   - **Effect Size Calculation:**
      - Cohen's d computed as (mean_beta - 0) / standard_deviation
      - Quantifies practical significance of connectivity strength
   **Output Saved:**
      - Excel files with `_ttest.xlsx` suffix containing:
         - Raw and FDR-adjusted p-values
         - T-statistics and Cohen's d effect sizes
         - Mean beta coefficients and standard deviations per channel
   - **Interpretation:**
      - FDR-adjusted p < 0.05: Significant functional connectivity between seed and target channel
      - Cohen's d > 0.8: Large effect size indicating strong connectivity
      - Positive/negative betas indicate activation/deactivation patterns

4. **Visualization in Satori through CMP file**
   - From folder: `SBA t-test` execute `#3 ttest cmp.py`
   - **Purpose:** Convert group-level t-test results from Excel format to Satori CMP (Color Map) files for 3D brain visualization of seed-based connectivity patterns
   - **Statistical Thresholding (Scaled Maps Only):**
      - Applies FDR-corrected p-value threshold (p ≤ 0.05) 
      - Sets non-significant t-values to zero for cleaner visualization
   - **Seed Channel Handling (Scaled Maps Only):**
      - Seed channels may show extremely high t-values (potentially infinite due to perfect self-correlation)
      - Script replaces maximum t-values (assumed to be seed channels) with second maximum value
      - Prevents visualization problems
   - **Normalization for Visualization (Scaled Maps Only):**
      - T-statistics are normalized to range [-1, +1] while preserving zero as anchor point
      - Normalization divides by maximum absolute value to maintain relative magnitudes
   - **Visualization Maps Created:**
      - **Raw T-statistic map:** Original statistical values 
      - **Cohen's d map:** Effect size magnitude
      - **Scaled T-statistic map:** FDR-thresholded, normalized values optimized for visual comparison
   **Output Saved:**
      - `{filename}_tvalues.cmp`: Raw T-statistic brain map for significance visualization
      - `{filename}_CohensD.cmp`: Effect size brain map for connectivity strength
      - `{filename}_tvalues_scaled.cmp`: FDR-thresholded, normalized T-statistic map for cross-method comparison
   - **Usage:** Load .cmp files into Satori

### SBA using GLM with Respiratory Predictor (SBA-GLM-Resp)
   - Same procedure with `SBA-GLM`
   - Added Respiratory Confounds in the GLM from Satori along with seed channels

### SBA using Correlation (SBA-Corr)

**Purpose:** Perform RSFC analysis with Correlation of seed channel with all other channels

*Script Folder:* `SBA Corr`

1. **Whole-Brain Correlation**
   - execute script: `#1a CrossCorrelationPerSubject.py`
   - Performs channel wise correlation for HbO/HbR
   - Saves the output in a single excel per subject with two sheets per heme group (HbO/HbR)
   - Excel suffix: `_CrossCorrelation.xlxs`

2. **Extract Seed Correlations**
   - execute script: `#1b_SBACorr_GroupedResults.py`
   - Extract seed correlation results from individual Excel files (e.g., `P01_CrossCorrelation.xlsx`)
   - Aggregates correlation values for predefined seed channels across all participants
   - **Seeds used:** HbO = S10-D7, HbR = S25-D23 (based on motor cortex GLM results)
   - **Output:** Two consolidated Excel files:
      - `SBA_Corr_Oxy_GroupedResults_S10D7.xlsx`: HbO seed correlations for all subjects
      - `SBA_Corr_Deoxy_GroupedResults_S25D23.xlsx`: HbR seed correlations for all subjects
   - Structure: Channels as rows, participants as columns

3. **Perform t-test:**
   - execute script: `#1c_SBA_Corr_ttest.py`
   - **Purpose:** Determine whether specific fNIRS channels show consistent correlation with the seed channel across multiple participants
   - **Statistical Method:**
      - One-sample t-test performed on correlation values (r-values) for each channel across participants
      - Null hypothesis (H₀): No significant correlation with seed channel (r = 0)
      - Alternative hypothesis (H₁): Consistent correlation with seed channel (r ≠ 0)
   **Output Saved:**
      - Excel files with `_ttest.xlsx` suffix containing:
         - Raw and FDR-adjusted p-values
         - T-statistics and Cohen's d effect sizes
         - Mean correlation coefficients and standard deviations per channel

4. **Visualize Results in Satori:**
   - execute script: `#1d_SBA_Corr_CMP.py`
   - **Purpose:** Convert group-level t-test results from Excel format to Satori CMP
   - **Special Handling for Infinite Values:**
      - Seed channels show perfect self-correlation (r = 1.0, std = 0)
      - Results in infinite t-statistics and Cohen's d values
      - Script replaces infinite values with maximum finite value for proper visualization
   - **Statistical Thresholding (Scaled Maps Only):**
      - Applies FDR-corrected p-value threshold (p ≤ 0.05) 
      - Sets non-significant t-values to zero for cleaner visualization
      - Preserves statistical significance in brain maps
   - **Normalization for Visualization (Scaled Maps Only):**
      - T-statistics are normalized to range [-1, +1] while preserving zero as anchor point
      - Normalization divides by maximum absolute value to maintain relative magnitudes
   - **Visualization Maps Created:**
      - **Raw T-statistic map:** Original raw t-values
      - **Cohen's d map:** Effect size magnitude for connectivity strength assessment
      - **Scaled T-statistic map:** FDR-thresholded, normalized values optimized for visualization
   **Output Saved:**
      - `{filename}_tvalues.cmp`: Raw T-statistic brain map 
      - `{filename}_CohensD.cmp`: Effect size brain map
      - `{filename}_tvalues_scaled.cmp`: FDR-thresholded, normalized T-statistic map for cross-method comparison
   - **Usage:** Load .cmp files into Satori to visualize 

## 4. ICA 

**Purpose:** Perform PCA and FastICA on preprocessed fNIRS resting-state data to generate Independent Component (IC) spatial maps, following Zhang et al. (2010) methodology with methodological improvements.

*Script Folder:* `ICA`

### 4.1 Individual-Level ICA Analysis
   - execute script: `#1_PCA_ICA_PIPELINE.py`
   - **Methodological Improvements over Zhang et al. (2010):**
      - Multiple FastICA runs with different random seeds for stability
      - Reproducible seeding strategy (fixed starting point + incremental seeds)
      - Robust IC selection using spatial correlation with motor task GLM maps
      - Sign alignment correction (ICA is sign-agnostic)
      - Median-based IC averaging across runs for enhanced stability

   **Steps of the script**
   1. **Data Loading and Preparation**
      - Load preprocessed SNIRF files with `_SatPreP` suffix
      - **Second-order polynomial detrending** applied to remove slow drifts

   2. **Dimensionality Reduction (PCA)**
      - Apply PCA separately to HbO and HbR signals
      - Retain components explaining 99% of variance
      - Reduces computational load while preserving signal information

   3. **Independent Component Analysis (FastICA)**
      - **Multiple runs approach:** 50 runs per subject with different random seeds
      - **Seeding strategy:** Fixed starting point (2025) + incremental seeds for reproducibility
      - **FastICA Parameters (estimating Zhang et al. 2010):**
         - Iterations: 10,000 (max_iter=10000)
         - Algorithm: "deflation" 
         - Nonlinearity functions: "skew" (g(u)=u²) and "logcosh" (alternative)
         - Tolerance: 1e-6 for convergence stability

   4. **Spatial Map Reconstruction**
      - Matrix multiplication: `A = pca.components_.T @ ica_fit.mixing_`
      - Recover IC spatial patterns in original channel space
      - Z-score normalization of spatial maps for standardization
      - **Code Explanation:**
         - `pca.components_.T`: Channel contributions to each PC
         - `ica_fit.mixing_`: IC contributions from each PC
         - `A`: Final spatial maps (channels × ICs)

   5. **Data Organization and Export**
      - Generate DataFrames: channels as rows, ICs as columns
      - Separate processing for HbO and HbR chromophores
      - Export to Excel with organized sheet structure

   **Input Requirements:**
   - Preprocessed SNIRF files from `RestingState_SatPreP_forICA` folder
   - Data must be: converted to concentration changes, uniform duration, steady-state trimmed, bandpass filtered (0.01-0.2 Hz), 1st order trends removed, z-transformed

   **Output Files:**
   - Excel files per subject per run: `[SubjectID]_[function]_Run[#].xlsx`
   - Two sheets per file: 'HbO Spatial Maps' and 'HbR Spatial Maps'
   - **Folder structure:** `1_FastICA_SM_Seed[starting_point]_Runs[n_runs]/[Subject]/`

   **Processing Statistics:**
   - **Total runs:** 31 subjects × 50 runs × 2 functions = 3,100 FastICA runs
   - **Estimated time:** ~47.5 hours total processing time
   - **Note:** logcosh (~80 min per subject) function is computationally slower than skew (~3min per subject)

### 4.2 Group-Level ICA Analysis
   - execute script: `#2_group_RSFC_ICA.py`
   - **Purpose:** Identify motor-related independent components for each subject and compute group statistics to determine brain regions showing consistent connectivity patterns across the group

   **Steps of the script**
   1. **IC Selection Strategy**
      - For each subject, for each run: correlate all ICs with group-level motor task T-values
      - Select IC with highest correlation (lowest p-value) per run
      - **Sign alignment:** Correct IC polarity to match motor reference map
      - Store best ICs across all runs per subject

   2. **Quality Control Procedures**
      - Channel name validation between IC data and motor reference maps
      - Short-distance channel removal (detector > 28)
      - Sign alignment verification before group analysis
      - Stability assessment across multiple runs

   3. **Two Analysis Approaches**
      - **Approach A - Highest Correlation:** Select single best IC across all runs per subject
      - **Approach B - Median Stability:** Calculate median IC across all runs per subject for enhanced stability

   4. **Group-Level Statistical Analysis**
      - One-sample t-test across subjects (H₀: mean = 0)
      - **Multiple Comparison Correction:** FDR correction using Benjamini-Hochberg method
      - **Effect Size Calculation:** Cohen's d for practical significance
      - Separate analysis for both nonlinearity functions (skew, logcosh)

   **Input Requirements:**
   - Excel files from Individual-Level ICA Analysis (`#1_PCA_ICA_PIPELINE.py`)
   - Motor task GLM group results for IC selection reference
   - **Folder structure:** `1_FastICA_SM_Seed[starting_point]_Runs[n_runs]/[Subject]/`

   **Output Files:**
   - **Per subject best ICs:** Excel files with best ICs per run (`[SubjectID]_bestICs.xlsx`)
      - Two sheets per file: 'Oxy Best ICs' and 'Deoxy Best ICs'
      - **Folder:** `6_AllRunsBestICs_PerSubject/`
   - **Group analysis results:** Excel files with T-values, P-values, FDR-corrected P-values, mean values, standard deviations, and Cohen's d
      - **Highest correlation approach:** `Group_RSFC_Hcor_[function].xlsx`
      - **Median-based approach:** `Group_RSFC_Med_[function].xlsx`
      - **Folders:** `7a_IC_HighestCorr_AcrossRuns/` and `7b_IC_Median_AcrossRuns/`
   - **Individual IC consolidation:** Excel files with selected ICs per subject
      - **Highest correlation:** `IC_IndCorr_[function].xlsx` and `IC_GrCorr_[function].xlsx`
      - **Median-based:** `IC_IndMed_[function].xlsx` and `IC_GrMed_[function].xlsx`

   **Key Methodological Advantages:**
   - Objective motor-relevance criteria for IC selection (no visual inspection)
   - Enhanced stability through multiple-run averaging (median approach)

5. **Visualize Results in Satori:**
   - Execute `#3_ICA_CMP.py`
   - **Purpose:** Convert group-level ICA resting-state functional connectivity (RSFC) results from Excel format to Satori CMP for visualization
   - **Spatial Mapping:**
      - Maps processed ICA channels to complete SATORI 134-channel layout
      - Handles missing channels (short channels removed during ICA preprocessing)
      - Creates master dataframes with standardized channel indexing
   - **Normalization & Thresholded for Visualization (Scaled Maps Only):**
      - T-statistics are first normalized to range [-1, +1] while preserving zero as anchor point
      - Then FDR thresholding is applied to show only significant patterns
   - **Visualization Maps Created:**
      - **Raw T-statistic map:** Original t values
      - **Cohen's d map:** Effect size magnitude
      - **Scaled T-statistic map:** Normalized then FDR-thresholded values optimized for visual comparison
   **Output Saved:**
   For each input Excel file, generates 6 CMP files:
   - HbO Maps:
      - `{filename}_tvalues_oxy.cmp`: Raw T-statistic 
      - `{filename}_CohensD_oxy.cmp`: Effect size 
      - `{filename}_tvalues_scaled_oxy.cmp`: Normalized, FDR-thresholded T-statistic map
   - HbR Maps:
      - `{filename}_tvalues_deoxy.cmp`: Raw T-statistic 
      - `{filename}_CohensD_deoxy.cmp`: Effect size
      - `{filename}_tvalues_scaled_deoxy.cmp`: Normalized, FDR-thresholded T-statistic map
   - **Usage:** Load .cmp files into Satori to visualize ICA-derived connectivity networks on 3D brain model

## 5. Performance Evaluation

### ROC Curves
**Purpose:** Evaluate the performance of different methods (e.g., SBA, ICA) using Receiver Operating Characteristic (ROC) analysis. 

*Script Folder:* Performance evaluation

1. **Data Organization**
   - Create folder `8_PerformanceEvaluation` 
   - Copy t-test results from all RSFC methods:
      - SBA-GLM: `Group_GLM_Betas_S##_D##_{chromophore}_ttest.xlsx`
      - SBA-GLM-Resp: Similar format with respiratory predictors
      - SBA-Corr: `SBA_Corr_{chromophore}_GroupedResults_S##D##_ttest.xlsx`
      - ICA: `Group_RSFC_{approach}_{function}.xlsx` (e.g., Med_logcosh, Hcor_skew)
   - Files renamed for processing convenience

2. **Execute `#1_All_ROC.py`**
   - **Golden Standards:**
      - **Motor Task Activation Map:** FDR-corrected (p ≤ 0.05) binarized motor cortex activation
      - **fOLD Binary Map:** Anatomically-defined motor regions (primary motor, premotor, secondary somatosensory)
   
   - **Data Processing:**
      - Normalizes effect sizes to 0-1 range using Min-Max scaling
      - Handles infinite values by replacing with second maximum finite value
      - Removes short-distance channels (detector > 28)
   
   - **ROC Analysis:**
      - Computes Area Under Curve (AUC) for all method combinations
   
   - **Output Files:**
      - **Figures:** `ROC_Curves_{golden_standard}.png` (300 DPI)
      - **Results:** `ROC_Results_{golden_standard}.txt` with detailed AUC rankings and statistics
      - **Folder:** `ROC_fOLD/` (automatically created)
   
   - **Visualization Features:**
      - Dual-panel plots (HbO vs HbR)
      - Distinct colors for each method using seaborn palette
      - Statistical summaries printed to console and saved to text files

**Interpretation:**
- **AUC = 0.5:** Random chance performance
- **AUC > 0.7:** Good discriminative ability
- **AUC > 0.8:** Excellent performance
- Higher AUC indicates better ability to distinguish motor-related connectivity patterns  

### Similarity of RSFC t-maps across methods

**Purpose:** Evaluate similarity between oxygenated (HbO) and deoxygenated (HbR) hemoglobin RSFC t-maps by spatial correlation of t-values across different methods.

*Script:* `#2_All_Symmetry.py`

**Theoretical Background:**
HbO and HbR RSFC t-maps should show the same network. Performing a spatial correlation allows us to how similar are the resulting RSFC motor map from the different heme signals.

**Methods Analyzed:**
- **SBA Methods:** SBA-GLM, SBA-GLM-Resp, SBA-Corr (separate HbO/HbR files)
- **ICA Methods:** ICA-Logcosh, ICA-Skew (single file with separate sheets)

**Processing Steps:**

1. **Data Organization**
   - Copy t-test results from all RSFC methods to `8_PerformanceEvaluation
      - already done from previous script

2. **Data Preprocessing**
   - Remove short-distance channels (detector > 28)
   - **SBA-specific:** Remove seed channels (S10-D7, S25-D23) to avoid self-correlation artifacts [handles infinite values in correlation data]
   - Channel alignment verification between HbO/HbR datasets

3. **Sign Alignment (ICA Methods Only)**
   - Verify alignment with motor task GLM reference maps
   - Apply HbR sign flip (`data_deoxy['T-Value'] = -data_deoxy['T-Value']`) to match HbO directionality; in effect avoid errors when min max normalizing for vizualization purposes

4. **Statistical Analysis**
   - Calculate Pearson spatial correlation between HbO and HbR t-values
   - Apply MinMax normalization [0,1] for visualization

5. **Visualization**
   - Generate square scatter plots with diagonal symmetry reference line
   - Correlation statistics displayed in legend

**Output Files:**
- **Scatter Plots:** `{Method}_HbR_HbO_Similarity.png` (300 DPI, square format)
- **Statistical Results:** `{Method}_HbR_HbO_Similarity.txt` with correlation coefficients and p-values
- **Folder:** `8_PerformanceEvaluation/Similarity/`

**Interpretation:**
- **Points near diagonal:** Good HbO-HbR similarity of RSFC t-maps



NOTE:

Note:
bvbabel library was used to read/write CMP files, and can be found
https://github.com/ofgulban/bvbabel


`Useful_Files` folder contains:
SATORI WORKFLOWS (for preprocessing and analysis)
   - RestingStateSCI.flow
   - MotorActionPreprocessing.flow
   - RestingStateSBA.flow
   - ICA_Preprocessing.flow
   - SBA_GLM_Oxy_S10D7.flow
   - SBA_GLM_Deoxy_S25D23.flow

Participant exclusion criteria excel: `Quality Assessment BioSignalBy JOAO PEREIRA.xlsx`
   - Joao Pereira assessed the quality of the respiratory signal (joaoandrefpereira@gmail.com)

fOLD BINARIZATION excel: `Channels to Brain Areas using fOLD.xlsx`
