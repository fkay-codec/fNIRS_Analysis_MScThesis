# DataHandling Folder

**Purpose:** Raw data processing, trimming, quality control, and participant exclusion

## Quick Script Reference:

**Data Organization:**
- `#1a_reorganize_snirf_files.py` → Reorganize Subject/Task → Task/Subject structure

**Automatic Processing:**
- `#2a_trimming_first_last.py` → Auto-detect & trim (RestingState/MotorAction)
- `#2b_Manual_trimming.py` → Handle problematic files (OverflowError/KeyError)
- `#2c_uniform_duration.py` → Sample-accurate uniform durations for group analysis

**Quality Control:**
- `#3a_findSCI.py` → Analyze SCI values, calculate exclusion percentages
- `#3b_exclude_reorganize.py` → Remove excluded participants from dataset

## Execution Order:

`1a → 2a → 2b → 2c → [Satori: OD conversion + SCI analysis] → 3a → [Manual QA] → 3b`


## Key Outputs:
- **Trimmed data:** `_TrimmedToEvents.snirf` (Resting) / `_TrimmedToEventsStSt.snirf` (Motor)
- **Uniform data:** `_OD_UnifDurat.snirf` 
- **Quality metrics:** `SCI07_PerParticipant.xlsx`
- **Final dataset:** Clean data with excluded participants removed

## Error Handling:
Problematic files flagged with suffixes: `_ProbOverflow`, `_ProbKeyError`, `_ProbNoAnnot`, `_ProbInvTim`

---
**For detailed documentation, see main README.md**