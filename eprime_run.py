from eprime_funcs import process_event, get_eventname
import argparse
import os
import pandas as pd

def main(args):

    # Load all files
    files = [os.path.join(args.data_dr, file) for file in os.listdir(args.data_dr)]

    # Make sure all text files
    assert len(files) == len([f for f in files if f.endswith('.txt')])

    # Grab unique events
    u_events = list(set([get_eventname(file) for file in files]))
    
    # Process each event
    dfs = []
    for event in u_events:
        print(f'Processing event: {event}')
        dfs.append(process_event(event, files))

        print()
        print('-----')
        print()
    
    # Concat all events
    df = pd.concat(dfs)

    # Then save
    df.to_csv(args.save_loc)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dr', type=str, help='Location of the top level directory where the processed SST event data is stored.')
    parser.add_argument('save_loc', type=str, help="Where to store the output csv.")
    args = parser.parse_args()

    main(args)