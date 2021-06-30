
Python code designed to generate some flags and other files of interest from already processed SST event level data.

To run this program, first clone the directory locally. Next, make sure you have python3+ installed.
This program just requires the external library pandas
which can be installed with `pip install pandas`. Next, the script run.py can be used. run.py takes the following two positional arguments:

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

