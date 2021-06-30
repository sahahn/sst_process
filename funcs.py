import os
import pandas as pd
import numpy as np
import warnings

def load_merged_df(dr):
    
    files = os.listdir(dr)
    files = [os.path.join(dr, f) for f in files]

    dfs = []
    for file in files:
        subj = '_'.join(file.split('/')[-1].split('_')[:2])
        df = pd.read_csv(file)
        df['src_subject_id'] = subj
        dfs.append(df)

    df = pd.concat(dfs, sort=True, axis=0)
    return df

def check_subj(row):
    if not isinstance(row['subject'], str):
        return False
    if row['subject'] in row['src_subject_id']:
        return True
    return False

def get_issue3_cnt(df):
    return len(df[(df['sst_ssd_rt'] < 50) & (df['sst_ssd_rt'] > 0) & (df['sst_ssd_dur'] <= 50)])

def zero_SSD_count(df):
    return len(df.loc[df['sst_ssd_dur'] == 0])

def get_issue3_df(df, eventname):
    
    group = df.groupby('subject')
    issue3_cnts = group.apply(get_issue3_cnt)
    issue3_ids = issue3_cnts[issue3_cnts > 0].index
    
    return_df = pd.DataFrame()
    return_df['glitch_subjs'] = issue3_ids
    return_df['eventname'] = eventname
    
    return return_df

def get_zero_ssd_df(df, eventname):
    group = df.groupby('subject')
    zero_cnt = group.apply(zero_SSD_count)
    zero_cnt_ids = zero_cnt[zero_cnt > 20].index
    
    return_df = pd.DataFrame()
    return_df['too_many_0SSD'] = zero_cnt_ids
    return_df['eventname'] = eventname
    
    return return_df

def get_mean_overt_go_rt(df):
    
    overt_go_trials = df[(df['sst_expcon'] == 'GoTrial') & (~df['sst_primaryrt'].isnull())]
    return np.mean(overt_go_trials['sst_primaryrt'])

def get_mean_failed_stop_rt(df):
    
    failed_stop_trials = df[(df['sst_expcon'] == 'VariableStopTrial') & (~df['sst_primaryrt'].isnull())]
    return np.mean(failed_stop_trials['sst_primaryrt'])

def get_violators_df(df, eventname):
    group = df.groupby('subject')
    
    go_rts = group.apply(get_mean_overt_go_rt)
    stop_rts = group.apply(get_mean_failed_stop_rt)

    violators = stop_rts[stop_rts > go_rts].index
    
    return_df = pd.DataFrame()
    return_df['violators'] = violators
    return_df['eventname'] = eventname
    
    return return_df

def calc_ssrt(s_df):
    
    # Separate just go trials
    go_trials = s_df.query('sst_expcon == "GoTrial"')
    
    # Set omissions to max go.rt - if primary rt is_null means omission
    go_trials[go_trials['sst_primaryrt'].isnull()] = go_trials.sst_primaryrt.max()
    
    # Sort the go trials
    sorted_go = go_trials.sst_primaryrt.sort_values(ascending=True)
    
    # Separate stop trials
    stop_trials = s_df.query('sst_expcon == "VariableStopTrial"')
    
    # Calc prob stop failure, where if sst_primaryrt is null, means correct no response
    prob_stop_failure = (1-stop_trials['sst_primaryrt'].isnull().mean())
    
    # Calc go.rt index
    index = prob_stop_failure * len(sorted_go)
    
    # If prob_stop_failure is 1, use the max go.rt
    if np.ceil(index) == len(sorted_go):
        index = len(sorted_go)-1
    else:
        index = [np.floor(index), np.ceil(index)]
    
    # Calc SSRT
    mean_ssd = np.mean(stop_trials['sst_ssd_dur'])
    
    try:
        ssrt = sorted_go.iloc[index].mean() - mean_ssd
    except IndexError:
        print('index = ', index)
        print('len sorted go =', len(sorted_go))
        print('prob stop fail =', prob_stop_failure)
        ssrt = np.NaN

    return ssrt

def get_ssrt_df(df, eventname):
    
    group = df.groupby('subject')
    ssrt = group.apply(calc_ssrt)
    ssrt_df = pd.DataFrame()

    ssrt_df['tfmri_sst_all_beh_total_issrt'] = ssrt
    ssrt_df['eventname'] = eventname

    return ssrt_df

def get_issue3_mask(df, eventname):
    group = df.groupby('subject')
    issue3_mask = group.apply(get_issue3_cnt)
    issue3_mask[issue3_mask > 0] = 1
    
    d = pd.DataFrame()
    d['tfmri_sst_beh_glitchflag'] = issue3_mask
    d['eventname'] = eventname
    return d

def get_zero_ssd_count(df, eventname):
    group = df.groupby('subject')
    zero_cnt = group.apply(zero_SSD_count)
    
    d = pd.DataFrame()
    d['tfmri_sst_beh_0SSDcount'] = zero_cnt
    d['eventname'] = eventname
    
    return d

def get_violators_mask(df, eventname):
    group = df.groupby('subject')
    go_rts = group.apply(get_mean_overt_go_rt)
    stop_rts = group.apply(get_mean_failed_stop_rt)

    violators = (stop_rts > go_rts)
    
    d = pd.DataFrame()
    d['tfmri_sst_beh_violatorflag'] = violators.astype(int)
    d['eventname'] = eventname

    return d

def process_event(event, event_dr, _print=print):

    _print(f'Loading and processing event: {event}')
    
    # Load df
    df = load_merged_df(event_dr)
    print(f'Found {len(df.src_subject_id.unique())} unique subjects.')
    
    # Check for mis-formatted subjects
    df['sub_okay'] = df[['subject', 'src_subject_id']].apply(check_subj, axis=1)
    weird = len(df[~df["sub_okay"]].subject.unique())
    if weird > 0:
        warnings.warn(f'{weird} subjects found where the saved subject' 
                      'column does not match the name of the file - or is not a valid str!')

    # Get a single dataframe with glitched subjects
    glitch_df = get_issue3_df(df, eventname=event)
    _print(f'Found {len(glitch_df)} glitched subjects.')
    
    # Get over 20 0 SSD df
    zero_ssd_df = get_zero_ssd_df(df, eventname=event)
    _print(f'Found {len(zero_ssd_df)} subjects with > 20 0-SSD trials.')

    # Get violaters df
    vio_df = get_violators_df(df, eventname=event)
    _print(f'Found {len(vio_df)} violators.')

    # Get SSRT df
    _print('Calculating SSRT')
    ssrt_df = get_ssrt_df(df, eventname=event)
    
    # Create full df with flags
    _print('Generate return formatted DataFrame')
    return_df = ssrt_df.copy()

    for func in [get_issue3_mask, get_zero_ssd_count, get_violators_mask]:
        return_df = pd.merge(return_df, func(df, event),
                             on=['subject', 'eventname'])

    # Keep just columns of interest
    return_df = return_df[['eventname', 'tfmri_sst_beh_glitchflag', 'tfmri_sst_beh_0SSDcount',
                           'tfmri_sst_beh_violatorflag', 'tfmri_sst_all_beh_total_issrt']]

    # Rename index
    return_df.index.name = 'src_subject_id'

    _print('--------------------------------------')
    return return_df, glitch_df, zero_ssd_df, ssrt_df
 



    


