import pandas as pd

import argparse
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Combine all labels.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('input_folder', type=str, help='Absolute path to folder with label csvs (SLASH INCLUDED).')

    args = parser.parse_args()

    train_oulu_path = args.input_folder + "protocol1_train.csv"
    celeba_path = args.input_folder + "celebA_final.csv"

    train_oulu_df = pd.read_csv(train_oulu_path)
    celeba_df = pd.read_csv(celeba_path)
    
    train_out_df = pd.concat([train_oulu_df, celeba_df], ignore_index=False, sort=False)

    train_out_df.to_csv(args.input_folder + "train.csv", index=False)
