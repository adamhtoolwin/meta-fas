import pandas as pd

import argparse
import os
from tqdm import tqdm
from PIL import Image

"""
Script to randomly sample only a fraction of the original csvs
"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Prune celeba data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('input_csv', type=str, help='File to check.')
    parser.add_argument('-lf', '--live-fraction', type=float, help='Fraction of live originals to sample.', default=0.05)
    parser.add_argument('-sf', '--spoof-fraction', type=float, help='Fraction of spoof originals to sample.', default=0.05)
    parser.add_argument('--debug', dest='debug', action='store_true', help='Debug mode to run once and see csv.')

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    out_data = {
        'path': [],
        'target': []
    }

    live_df = df.loc[df['target'] == 1]
    spoof_df = df.loc[df['target'] == 0]

    live_final = live_df.sample(frac=args.live_fraction)
    spoof_final = spoof_df.sample(frac=args.spoof_fraction)

    final_df = pd.concat([live_final, spoof_final], ignore_index=True, sort=False)

    print("Using %f of lives (%d) and %f of spoofs (%d)" % (args.live_fraction, len(live_df), args.spoof_fraction, len(spoof_df)))
    print("Final size: % d" % len(final_df))
    final_df.to_csv("/root/datasets/CelebA_Spoof/metas/intra_test/celebA_final_pruned.csv", index=False)
