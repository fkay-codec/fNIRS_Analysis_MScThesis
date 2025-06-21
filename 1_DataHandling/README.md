### Data Handling Folder

'''1_reorganize snirf files.py'''
In the first script, I reorganized the data to be extracted from their original directory
e.g.: Subject --> RestingState --> .snirf file

and be reorganized per task:
e.g. RestingState --> subject_number_resting_state.snirf

2_findSCI.py
In the second script, I read the excel file with the Scalp Coupling Index per subject and compute the number/percentage of rejected channels (long/short) per subject, based on SCI = 0.7; then save the output in an excel file