Python code designed to generate some flags and other files of interest from already processed SST event level data.

To run this program, first clone the directory locally. Next, make sure you have python3+ installed.
This program just requires the external library pandas
which can be installed with `pip install pandas`.

# From E-Prime Level Data (.txt)

This is option is for generating the flags of interest from e-prime level minimally processed text files.
The python script `eprime_run.py` should be used which accepts two positional arguments:

1. data_dr

     The location of the folder storing all of the eprime files. All different events should be present in the same
     folder, e.g. consider the example folder SST_EPRIME:

      - SST_EPRIME/NDAR_INVXXXXXXXX_2_year_follow_up_y_arm_1_fMRI_SST_task_20200229140741.txt
      - SST_EPRIME/NDAR_INVXXXXXXXX_baseline_year_1_arm_1_fMRI_SST_task_20180310115631.txt
      - SST_EPRIME/NDAR_INVXXXXXXXX_4_year_follow_up_y_arm_1_fMRI_SST_task_20210415120352.txt
      - SST_EPRIME/...

2. data_loc
   
   The file location in which to save a .csv output file containing the columns: 

    - 'src_subject_id'
    - 'tfmri_sst_all_beh_total_issrt',
    - 'tfmri_sst_beh_glitchflag',
    - 'tfmri_sst_beh_glitchcnt',
    - 'tfmri_sst_beh_0SSDcount',
    - 'tfmri_sst_beh_0SSD>20flag',
    - 'tfmri_sst_beh_violatorflag',
    - 'eventname'

This script performs a number of different things to the raw data:

- Loads the files, handling inconsistent naming of columns.
- Performs a number of checks to make sure the internal saved name and file names align - There are hundreds of cases where they do not, so when in doubt the subject id associated with the filename is used.
- Truncates the files internally to just the columns of interest.
- Removes the NaN and fix trials from the start and end
- Handles duplicates (same subject at same eventname) by checking to see if one of the duplicates contained a different internal reference to another valid subject name. Checks to see if one of the files has the full 360 trials and the other doesn't (in that case dropping the incomplete copy). Otherwise, any duplicates are dropped.
- Drops any subjects without complete data for both runs (a total of 360 trials)
- Fixes inconsistencies where in some files trial_type is saved under 'Procedure[SubTrial]' and in other it is saved under 'Procedure[Trial]', in others just 'Procedure'. Likewise fixes inconsistent internal naming where sometimes 'StopTrial' is saved under a variation of 'VariableStopTrial'.
- Fixes inconsistent coding with respect to how columns Go.RESP, Go.CRESP, Fix.RESP, StopSignal.RESP and SSD.RESP are coded (e.g., some are saved as '1,{LEFTARROW}' and others as just '1' or '1.0' or '1{LEFTARROW}').
- Fixes the calculation of a response is a correct_go_response and a correct_stop_response according to the updated coded of the flags.
- Calculates the adjusted correct go rt correctly for go responses made during the Fix period.
- Calculates the adjusted correct stop rt correctly for stop responses made during the Fix period, stop responses made during the SSD period and for stop responses made during the stop period.
- Any subjects without any recorded go responses, e.g., all omissions, are dropped.
- Lastly, the different flags and other measures (e.g. ssrt) are calculated from this processed data.

# Already Processed Data (.csv)

This option is for generating the flags of interest from already highly processed data made available from the TLBD group.
The script run.py can be used. run.py takes the following two positional arguments:

1. data_dr

    This should be the path of the folder containing the names of event specific folders: e.g. given a folder like below, SST_Proc should be passed:

    - SST_Proc/baseline_year_1_arm_1/NDAR_INVXXXXXXX_baseline_year_1_arm_1_sst.csv
    - SST_Proc/baseline_year_1_arm_1/NDAR_INVYYYYYYY_baseline_year_1_arm_1_sst.csv
    - SST_Proc/baseline_year_1_arm_1/...
    - SST_Proc/2_year_follow_up_y_arm_1/NDAR_INVXXXXXXX_2_year_follow_up_y_arm_1_sst.csv
    - SST_Proc/2_year_follow_up_y_arm_1/NDAR_INVYYYYYYY_2_year_follow_up_y_arm_1_sst.csv
    - SST_Proc/2_year_follow_up_y_arm_1/...

2. save_dr: The location of the directory in which to save the output csv files. Note this folder will be created if it does not
   already exist. The following output files will be generated in this folder when the script is complete:
   
   - save_dr/new_columns.csv:
    
     A csv with columns 'src_subject_id', 'eventname', 'tfmri_sst_beh_glitchflag', 'tfmri_sst_beh_0SSDcount', 'tfmri_sst_beh_violatorflag', 'tfmri_sst_all_beh_total_issrt'.
  
   - save_dr/glitch_subjects.csv
  
        A csv with the subject ids and eventnames for the glitched subjects.
  
   - save_dr/over_20_ssd_subjects.csv

        A csv with the subject ids and eventnames for any subjects with > 20 SSD trials

   - save_dr/ssrt.csv

        A csv with the newly calculated SSRT, subject_id and eventname for all subjects.

The last optional parameter is adding the flag --quiet, which will mute automatic verbose statements.

Example usage:

```
python run.py /home/sage/SST_Proc output/
```

With no verbosity:

```
python run.py /home/sage/SST_Proc output/ --quiet
```

