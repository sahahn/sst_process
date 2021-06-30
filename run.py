import argparse
import os
import pandas as pd
from funcs import process_event

def main(args):

    verbose = ~args.quiet

    def _print(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    events = os.listdir(args.data_dr)
    event_drs = [os.path.join(args.data_dr, event) for event in events]
    _print(f'Found eventnames / subfolders: {events}')
    
    # Process each event seperately
    return_dfs, glitch_dfs, zero_ssd_dfs, ssrt_dfs = [], [], [], []
    for event, event_dr in zip(events, event_drs):
        return_df, glitch_df, zero_ssd_df, ssrt_df =\
            process_event(event, event_dr, _print=_print)
        
        # Store
        return_dfs.append(return_df) 
        glitch_dfs.append(glitch_df)
        zero_ssd_dfs.append(zero_ssd_df)
        ssrt_dfs.append(ssrt_df)
 
    # Concat all
    return_df = pd.concat(return_dfs)
    glitch_df = pd.concat(glitch_dfs)
    zero_ssd_df = pd.concat(zero_ssd_dfs)
    ssrt_df = pd.concat(ssrt_dfs)

    # Save
    os.makedirs(args.save_dr, exist_ok=True)
    _print(f'Saving to {args.save_dr}')
    return_df.to_csv(os.path.join(args.save_dr, 'new_columns.csv'))
    glitch_df.to_csv(os.path.join(args.save_dr, 'glitch_subjects.csv'), index=False)
    zero_ssd_df.to_csv(os.path.join(args.save_dr, 'over_20_ssd_subjects.csv'), index=False)
    ssrt_df.to_csv(os.path.join(args.save_dr, 'ssrt.csv'), index=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dr', type=str, help='Location of the top level directory where the processed SST event data is stored.')
    parser.add_argument('save_dr', type=str, help="Where to store the output, this directory will be created if it doesn't already exist")
    parser.add_argument('--quiet', action='store_const', const=True, default=False, help='Mute verbose statements')
    args = parser.parse_args()

    main(args)