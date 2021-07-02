import pandas as pd
import numpy as np

class Subj():

    def __init__(self, name, df, file=None):

        self.name = name
        self.df = df
        self.file = file

    @property
    def full_name(self):
        return 'NDAR_INV' + self.name

    def __lt__(self, other):
        return self.name < other.name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f'{self.name} - {self.df.shape}'


def get_eventname(file):
    return '_'.join(file.split('/')[-1].split('_')[2:-4])


def get_subj(file, full=False):

    subj = '_'.join(file.split('/')[-1].split('_')[:2])

    if full:
        return subj

    return subj.replace('NDAR_INV', '')


def remove_nan_fix_trials(df):

    df = df[np.logical_and(df.TrialCode != 'BeginFix',
                           df.TrialCode != 'EndFix')]

    df = df[~df['TrialCode'].isnull()]

    return df


def edit_distance(first, second):

    matrix = np.zeros((len(first)+1, len(second)+1), dtype=np.int)

    for i in range(len(first)+1):
        for j in range(len(second)+1):
            if i == 0:
                matrix[i][j] = j
            elif j == 0:
                matrix[i][j] = i
            else:
                matrix[i][j] = min(matrix[i][j-1] + 1,
                                   matrix[i-1][j] + 1,
                                   matrix[i-1][j-1] + 2 if first[i-1] != second[j-1] else matrix[i-1][j-1] + 0)

    return matrix[len(first)][len(second)]


def check_subject_issues(u_subjects_issue, name_issue):

    n_is_nan = 0
    n_reduced = 0
    n_lower = 0
    n_zero = 0
    n_bad_order = 0
    n_one_away = 0
    n_two_away = 0
    n_rest = 0

    worrying = {}

    for nar_guid, file in name_issue:
        subj_name = get_subj(file)

        if pd.isnull(nar_guid):
            n_is_nan += 1
            continue
        if len(subj_name) == len(nar_guid) and set(subj_name) == set(nar_guid):
            n_bad_order += 1
            continue
        if subj_name.endswith(nar_guid):
            n_reduced += 1
            continue

        # Change to upper case
        nar_guid = nar_guid.upper()

        if subj_name == nar_guid:
            n_lower += 1
            continue
        if len(subj_name) == len(nar_guid) and set(subj_name) == set(nar_guid):
            n_bad_order += 1
            n_lower += 1
            continue
        if subj_name.endswith(nar_guid):
            n_reduced += 1
            n_lower += 1
            continue

        # Change O to 0
        nar_guid = nar_guid.replace('O', '0')

        if subj_name == nar_guid:
            n_zero += 1
            continue
        if len(subj_name) == len(nar_guid) and set(subj_name) == set(nar_guid):
            n_bad_order += 1
            n_zero += 1
            continue
        if subj_name.endswith(nar_guid):
            n_reduced += 1
            n_zero += 1
            continue

        # Get edit distance
        dist = edit_distance(nar_guid, subj_name)
        if dist == 1:
            n_one_away += 1
            continue
        if dist == 2:
            n_two_away += 1
            continue

        if len(nar_guid) == 8 and dist > 4:
            worrying[file] = nar_guid
            continue

        n_rest += 1

    print(f'only lists NARGUID in the first row - {len(u_subjects_issue)}')
    print(f'missing (NaN) - {n_is_nan}')
    print(f'truncated - {n_reduced}')
    print(f'lower case - {n_lower}')
    print(f'0 coded as O - {n_zero}')
    print(f'jumbled order - {n_bad_order}')
    print(f'edit distance=1 - {n_one_away}')
    print(f'edit distance=2 - {n_two_away}')
    print(f'other misc errors - {n_rest}')
    print(f'reference to different subjects - {len(worrying)}')

    return worrying


def load_files(files):

    u_subjects_issue = []
    name_issue = []

    def load_file(file):
        '''Load a single run for a single subject'''

        used_cols = ['Procedure[SubTrial]', 'Procedure[Trial]', 'Go.RESP', 'Go.CRESP', 'Fix.RESP',
                     'StopSignal.RESP', 'SSD.RESP', 'SSD.RT', 'Stimulus',
                     'Go.RT', 'Go.Duration', 'Fix.RT', 'StopSignal.RT', 'StopSignal.Duration',
                     'SSDDur', 'NARGUID', 'TrialCode']

        # Load
        df = pd.read_csv(file, sep='\t', dtype={'NARGUID': np.object})

        # Sometimes named just procedure
        df = df.rename({'Procedure': 'Procedure[Trial]'}, axis=1)

        # If missing, fill with NaN
        if 'Procedure[SubTrial]' not in df:
            df['Procedure[SubTrial]'] = np.nan

        # Make just one subject name
        # If more than one, set to first
        u_subjects = df['NARGUID'].unique()
        if len(u_subjects) != 1:
            u_subjects_issue.append([u_subjects, file])
            u_subjects = [u_subjects[0]]

        # Make sure subj name matches file
        # If they don't go with file name
        subj_name = get_subj(file, full=False)
        if subj_name != u_subjects[0]:
            name_issue.append([u_subjects[0], file])
            df['NARGUID'] = subj_name

        # Remove fix trials
        df = remove_nan_fix_trials(df)

        # Only keep columns of interest
        return df[used_cols]

    # Load all files
    subjs = []
    for file in files:
        subjs.append(Subj(name=get_subj(file),
                          df=load_file(file),
                          file=file))

    # Check subject level issues
    worrying = check_subject_issues(u_subjects_issue, name_issue)

    print(f'Total found - {len(subjs)}')
    return subjs, worrying


def process_duplicates(subjs, worrying):

    n_renamed = 0

    # Process duplicate subjects
    u, cnts = np.unique(subjs, return_counts=True)
    removed = []

    for repeat in u[cnts > 1]:

        # Get the copies
        copies = [s for s in subjs if s == repeat]

        # Assume at most 2 copies
        assert len(copies) == 2

        # If also in worrying, assume that one was actually mis-coded
        # updates in place
        if any([copy.file in worrying for copy in copies]):
            for copy in copies:
                if copy.file in worrying:
                    copy.name = worrying[copy.file]
                    del worrying[copy.file]
                    n_renamed += 1

        # Check if just one has full 360, in that case use that one
        elif sum([copy.df.shape[0] == 360 for copy in copies]) == 1:
            for copy in copies:
                if copy.df.shape[0] != 360:
                    removed.append(copy)
                    subjs.remove(copy)

        # Final case is to remove both - could maybe recover more here,
        # i.e., may some are split into two runs?
        else:

            for copy in copies:
                removed.append(copy)
                subjs.remove(copy)

    print(f'Changed the names of files found to be misnamed - {n_renamed}')
    print(f'Removed due to unknown duplicate status - {len(removed)}')

    # Remove rest of worrying
    to_remove = []
    for subj in subjs:
        if subj.file in worrying:
            to_remove.append(subj)

    for subj in to_remove:
        subjs.remove(subj)

    print(
        f'Removed rest of remaining files with ref to dif subject - {len(to_remove)}')

    return subjs


def ensure_two_runs(subjs):

    n_removed = 0

    for subj in subjs:
        if subj.df.shape[0] != 360:
            subjs.remove(subj)
            n_removed += 1

    print(f'Removed subjects for != 360 trials - {n_removed}')
    return subjs


def set_trial_type(subj):
    '''Operates on each subject in place'''

    # Add new column
    subj.df['trial_type'] = subj.df['Procedure[SubTrial]']

    # Fill where missing
    missing = subj.df['trial_type'].isnull()
    subj.df.loc[missing, 'trial_type'] = subj.df.loc[missing,
                                                     'Procedure[Trial]']

    # Replace names
    subj.df['trial_type'] = subj.df['trial_type'].replace(
        'VariableStopTrial.*', 'StopTrial', regex=True)


def fix_resps(subj):
    '''Operates on each subject in place'''

    cresp_replace = {'2.0': 2.0,
                     '1.0': 1.0,
                     '3.0': 3.0,
                     '4.0': 4.0,
                     '1,{LEFTARROW}': 1.0,
                     '1{LEFTARROW}': 1.0,
                     '2,{RIGHTARROW}': 2.0,
                     '2{RIGHTARROW}': 2.0}

    resp_replace = {'2.0': 2.0,
                    '1.0': 1.0,
                    '3.0': 3.0,
                    '4.0': 4.0,
                    '{LEFTARROW}': 1.0,
                    '1{LEFTARROW}': 1.0,
                    '{RIGHTARROW}': 2.0,
                    '2{RIGHTARROW}': 2.0}

    subj.df['Go.RESP'].replace(to_replace=resp_replace, inplace=True)
    subj.df['Go.RESP'] = subj.df['Go.RESP'].astype('float')

    subj.df['Go.CRESP'].replace(to_replace=cresp_replace, inplace=True)
    subj.df['Go.CRESP'] = subj.df['Go.CRESP'].astype('float')

    subj.df['Fix.RESP'].replace(to_replace=resp_replace, inplace=True)
    subj.df['Fix.RESP'] = subj.df['Fix.RESP'].astype(float)

    subj.df['StopSignal.RESP'].replace(to_replace=resp_replace, inplace=True)
    subj.df['StopSignal.RESP'] = subj.df['StopSignal.RESP'].astype(float)

    subj.df['SSD.RESP'].replace(to_replace=resp_replace, inplace=True)
    subj.df['SSD.RESP'] = subj.df['SSD.RESP'].astype(float)


def set_correct_go(subj):
    '''Operates on each subject in place'''

    subj.df['correct_go_response'] = np.NaN

    subj.df.loc[(~subj.df['Go.RESP'].isnull()) &
                (subj.df['Go.CRESP'] == subj.df['Go.RESP']), 'correct_go_response'] = float(1)

    subj.df.loc[(subj.df['Go.RESP'].isnull()) &
                (subj.df['Go.CRESP'] == subj.df['Fix.RESP']), 'correct_go_response'] = float(1)

    subj.df.loc[(~subj.df['Go.RESP'].isnull()) &
                (subj.df['Go.CRESP'] != subj.df['Go.RESP']) &
                (subj.df['trial_type'] == 'GoTrial'), 'correct_go_response'] = float(0)

    subj.df.loc[(subj.df['Go.RESP'].isnull()) &
                (subj.df['Go.CRESP'] != subj.df['Fix.RESP']) &
                (subj.df['trial_type'] == 'GoTrial'), 'correct_go_response'] = float(0)

    subj.df.loc[(subj.df['Go.RESP'].isnull()) & (subj.df['Fix.RESP'].isnull()) &
                (subj.df['trial_type'] == 'GoTrial'), 'correct_go_response'] = 'omission'


def set_correct_stop(subj):
    '''Operates on each subject in place'''

    subj.df['correct_stop'] = np.NaN

    crt_stop_msk = ((subj.df['StopSignal.RESP'].isnull()) &
                    (subj.df['Fix.RESP'].isnull()) &
                    (subj.df['SSD.RESP'].isnull()) &
                    (subj.df['trial_type'] == 'StopTrial'))

    subj.df.loc[crt_stop_msk, 'correct_stop'] = float(1)

    inc_stop_msk = ((~(subj.df['StopSignal.RESP'].isnull()) |
                     ~(subj.df['Fix.RESP'].isnull()) |
                     ~(subj.df['SSD.RESP'].isnull())) &
                    (subj.df['trial_type'] == 'StopTrial'))
    subj.df.loc[inc_stop_msk, 'correct_stop'] = float(0)


def set_correct_go_rt(subj):

    go_fix_resp = (~subj.df['Fix.RESP'].isnull()) & \
                  (subj.df['Go.RESP'].isnull()) & \
                  (subj.df['trial_type'] == 'GoTrial')
    go_fix_idx = go_fix_resp[go_fix_resp].index

    subj.df['go_rt_adjusted'] = subj.df['Go.RT'].copy()
    subj.df.loc[go_fix_idx, 'go_rt_adjusted'] =\
        subj.df.loc[go_fix_idx]['Go.Duration'] + \
        subj.df.loc[go_fix_idx]['Fix.RT']


def set_correct_stop_rt(subj):

    # Init
    subj.df['stop_rt_adjusted'] = subj.df['StopSignal.RT']

    # Set to Stop signal duration + fix rt, as was answered during fix
    stop_fix_resp = (~subj.df['Fix.RESP'].isnull()) & \
                    (subj.df['StopSignal.RESP'].isnull()) & \
                    ((subj.df['trial_type'] == 'StopTrial')
                     & (subj.df['correct_stop'] == 0))
    stop_fix_idx = stop_fix_resp[stop_fix_resp].index

    subj.df.loc[stop_fix_idx, 'stop_rt_adjusted'] =\
        subj.df.loc[stop_fix_resp]['StopSignal.Duration'] + \
        subj.df.loc[stop_fix_resp]['Fix.RT']

    # Adjust for answers during SSD
    stop_SSD_resp = ~subj.df['SSD.RESP'].isnull()
    stop_SSD_resp_idx = stop_SSD_resp[stop_SSD_resp].index

    subj.df.loc[stop_SSD_resp_idx, 'stop_rt_adjusted'] =\
        subj.df.loc[stop_SSD_resp_idx, 'SSD.RT']

    # Set updated for response during stop
    stop_resp_mask = ((~(subj.df['StopSignal.RESP'].isnull()) | ~(subj.df['Fix.RESP'].isnull())) &
                      (~subj.df['stop_rt_adjusted'].isnull()))
    stop_resp_idx = stop_resp_mask[stop_resp_mask].index

    subj.df.loc[stop_resp_idx, 'stop_rt_adjusted'] =\
        subj.df.loc[stop_resp_idx, 'stop_rt_adjusted'] +\
        subj.df.loc[stop_resp_idx, 'SSDDur']


def get_glitched_cnt(subj):
    ''' Number of glitched trials'''
    df = subj.df
    return len(df[(df['SSD.RT'] < 50) & (df['SSD.RT'] > 0) & (df['SSDDur'] <= 50)])


def get_glitched_flag(subj):
    '''Flag is any at all'''
    
    df = subj.df
    return len(df[(df['SSD.RT'] < 50) & (df['SSD.RT'] > 0) & (df['SSDDur'] <= 50)]) > 0


def get_0SSD_cnt(subj):
    df = subj.df
    return len(df.loc[df['SSDDur'] == 0])


def get_0SSD_flag(subj):
    return get_0SSD_cnt(subj) > 20


def is_violator(subj):
    df = subj.df

    go_rt = np.nanmean(df.loc[(df['correct_go_response'] != 'omission') &
                              (df['trial_type'] == 'GoTrial')]['go_rt_adjusted'])
    stop_rt = np.mean(df.loc[df['correct_stop'] == 0]['stop_rt_adjusted'])

    return stop_rt > go_rt


def calc_ssrt(subj):

    # Grab copy
    s_df = subj.df

    # Separate just go trials
    go_trials = s_df.query('trial_type == "GoTrial"').copy()

    # Set omissions to max go.rt - if primary rt is_null means omission
    # then sort go trials
    go_trials.loc[go_trials['correct_go_response'] ==
                  "omission"] = go_trials.go_rt_adjusted.max()
    sorted_go = go_trials.go_rt_adjusted.sort_values(ascending=True)

    # Get stop trials + prob stop failure
    stop_trials = s_df.query('trial_type == "StopTrial"')

    # Calc prob stop failure
    prob_stop_failure = (1-stop_trials.correct_stop.mean())

    # Calc go.rt index
    index = prob_stop_failure * len(sorted_go)

    # If prob_stop_failure is 1, use the max go.rt
    if np.ceil(index) == len(sorted_go):
        index = len(sorted_go)-1
    else:
        index = [np.floor(index), np.ceil(index)]

    # Calc SSRT
    mean_ssd = np.mean(stop_trials['SSDDur'])

    try:
        ssrt = sorted_go.iloc[index].mean() - mean_ssd
    except IndexError:
        print(s_df.NARGUID.iloc[0], s_df.src_subject_id.iloc[0])
        print('index = ', index)
        print('len sorted go =', len(sorted_go))
        print('prob stop fail =', prob_stop_failure)
        ssrt = np.NaN

    return ssrt


def get_output_df(subjs, eventname):
    
    # Init df
    df = pd.DataFrame(index=[s.full_name for s in subjs],
                      columns=['tfmri_sst_all_beh_total_issrt',
                               'tfmri_sst_beh_glitchflag',
                               'tfmri_sst_beh_glitchcnt',
                               'tfmri_sst_beh_0SSDcount',
                               'tfmri_sst_beh_0SSD>20flag',
                               'tfmri_sst_beh_violatorflag',
                               'eventname'])
    
    # For each subject
    for subj in subjs:
        df.loc[subj.full_name, 'tfmri_sst_all_beh_total_issrt'] = calc_ssrt(subj)
        df.loc[subj.full_name, 'tfmri_sst_beh_glitchflag'] = get_glitched_flag(subj)
        df.loc[subj.full_name, 'tfmri_sst_beh_glitchcnt'] = get_glitched_cnt(subj)
        df.loc[subj.full_name, 'tfmri_sst_beh_0SSDcount'] = get_0SSD_cnt(subj)
        df.loc[subj.full_name, 'tfmri_sst_beh_0SSD>20flag'] = get_0SSD_flag(subj)
        df.loc[subj.full_name, 'tfmri_sst_beh_violatorflag'] = is_violator(subj)
    
    # Add event name
    df['eventname'] = eventname

    # Add index name
    df.index.name = 'src_subject_id'

    # Return
    return df


def check_omissions(subjs):

    n_removed = 0
    for subj in subjs:

        n_missing = len(subj.df.loc[(subj.df['Go.RESP'].isnull()) & (subj.df['Fix.RESP'].isnull()) &
                                  (subj.df['trial_type'] == 'GoTrial')])
        n_go_trials = len(subj.df.loc[subj.df['trial_type'] == 'GoTrial'])

        # Remove
        if n_missing == n_go_trials:
            subjs.remove(subj)
            n_removed += 1

    print(f'Subjects removed for all go trial omissions - {n_removed}')
    return subjs


def process_event(event, files):

    # Get just one events files
    e_files = [file for file in files if get_eventname(file) == event]
    print(f'Found files - {len(e_files)}')
    
    # Load files as dfs wrapped in Subj class
    print('Loading Files...')
    subjs, worrying = load_files(e_files)

    # Run twice, as renamed subjects can then overlap
    subjs = process_duplicates(subjs, worrying)
    subjs = process_duplicates(subjs, worrying)

    # Ensure two runs
    subjs = ensure_two_runs(subjs)
    print(f'Remaining subjects - {len(subjs)}')

    # Perform by subject processing
    print('Fixing data...')
    for subj in subjs:

        # Fix trial type
        set_trial_type(subj)

        # Fix responses
        fix_resps(subj)

        # Set correct go and stop
        set_correct_go(subj)
        set_correct_stop(subj)

        # Set correct go and stop rt
        set_correct_go_rt(subj)
        set_correct_stop_rt(subj)

    # Check for any subjects with all omissions on go trials
    print('Checking for any subjects with all omissions')
    subjs = check_omissions(subjs)
    print(f'Remaining subjects - {len(subjs)}')
    
    # Return output df
    return get_output_df(subjs, event)


