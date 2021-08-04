import pandas as pd

import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Combine provided HKBU metadata.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('input_folder', type=str, help='Absolute path to folder with label files (SLASH INCLUDED).')

    args = parser.parse_args()

    attack_path = args.input_folder + "attack.csv"
    real_path = args.input_folder + "real.csv"

    attack_df = pd.read_csv(attack_path)
    real_df = pd.read_csv(real_path)

    out_df = pd.concat([attack_df, real_df], ignore_index=False, sort=False)

    out_df.to_csv(args.input_folder + "test.csv", index=False)
