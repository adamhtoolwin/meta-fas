import pandas as pd

import argparse
import os
from tqdm import tqdm
from PIL import Image


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Check HKBU labels for missing files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('input_csv', type=str, help='File to check.')
    parser.add_argument('output_csv', type=str, help='Absolute file path to output csv.')
    parser.add_argument('--debug', dest='debug', action='store_true', help='Debug mode to run once and see csv.')

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    out_data = {
        'path': [],
        'target': []
    }

    print("Starting reading labels...")
    pbar = tqdm(range(len(df)))
    dropped_count = 0
    for index in pbar:
        path = df.at[index, 'path']
        target = df.at[index, 'target']
        # pbar.set_description("Processing image %s" % path)

        try:
            image = Image.open(path)
            
            out_data['path'].append(path)
            out_data['target'].append(target)
        except FileNotFoundError:
            dropped_count += 1

            if args.debug:
                break

    print("Finished - Dropped %d rows" % dropped_count)
    out_df = pd.DataFrame(out_data)

    print("Final size: % d" % len(out_df))
    out_df.to_csv(args.output_csv, index=False)
