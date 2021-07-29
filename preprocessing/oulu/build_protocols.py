import pandas as pd

import argparse
from tqdm import tqdm


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Build protocols from pregenerated oulu metadata.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('input_folder', type=str, help='Absolute path to folder with label files (SLASH INCLUDED).')
    parser.add_argument('--debug', dest='debug', action='store_true', help='Debug mode to run once and see csv.')

    args = parser.parse_args()

    train_path = args.input_folder + "train.csv"
    val_path = args.input_folder + "dev.csv"
    test_path = args.input_folder + "test.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True, sort=False)

    pbar = tqdm(combined_df.iterrows())
    val_indices = []
    for index, row in pbar:
        file_type = row['path'].split("/")[-2].split("_")[-1]
        person_id = int(row['path'].split("/")[-2].split("_")[-2])

        if file_type == '4' or file_type == '5':
            val_indices.append(index)

        if person_id >= 36 and person_id <= 55 and file_type == '1':
            val_indices.append(index)

        if args.debug:
            break

    protocol1_train = combined_df.iloc[~combined_df.index.isin(val_indices)]
    protocol1_val = combined_df.iloc[val_indices]

    pbar2 = tqdm(combined_df.iterrows())
    val_indices = []
    for index, row in pbar2:
        file_type = row['path'].split("/")[-2].split("_")[-1]
        person_id = int(row['path'].split("/")[-2].split("_")[-2])

        if file_type == '2' or file_type == '3':
            val_indices.append(index)

        if person_id >= 36 and person_id <= 55 and file_type == '1':
            val_indices.append(index)

        if args.debug:
            break

    protocol2_train = combined_df.iloc[~combined_df.index.isin(val_indices)]
    protocol2_val = combined_df.iloc[val_indices]

    protocol1_train.to_csv(args.input_folder + "protocol1_train.csv", index=False)
    protocol1_val.to_csv(args.input_folder + "protocol1_val.csv", index=False)

    protocol2_train.to_csv(args.input_folder + "protocol2_train.csv", index=False)
    protocol2_val.to_csv(args.input_folder + "protocol2_val.csv", index=False)
