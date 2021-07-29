import pandas as pd

import argparse
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Combine provided celebA metadata.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('input_folder', type=str, help='Absolute path to folder with label files (SLASH INCLUDED).')
    parser.add_argument('-p', '--prefix', type=str, required=True, help='Absolute path to append to image paths in output labels.')

    args = parser.parse_args()

    train_path = args.input_folder + "train_label.txt"
    test_path = args.input_folder + "test_label.txt"

    train_df = pd.read_csv(train_path, sep=" ", names=["path", "target"])
    test_df = pd.read_csv(test_path, sep=" ", names=["path", "target"])
    
    out_df = pd.concat([train_df, test_df], ignore_index=False, sort=False)
    out_df['path'] = args.prefix + out_df['path']
    
    out_df.to_csv(args.input_folder + "celebA.csv", index=False)
