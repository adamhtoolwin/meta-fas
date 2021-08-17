import pandas as pd

import argparse
from tqdm import tqdm


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Filter samples from only one lighting variation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('input_csv', type=str, help='Absolute path to csv with label files.')
    parser.add_argument('output_csv', type=str, help='Absolute path to output csv with label files.')
    parser.add_argument('--debug', dest='debug', action='store_true', help='Debug mode to run once and see csv.')

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    out_indices = []

    pbar = tqdm(df.iterrows())
    for index, row in pbar:
        lighting_variation = row['path'].split("/")[-2].split("_")[1]
        input_sensor = row['path'].split("/")[-2].split("_")[0]

        if lighting_variation == '01' and input_sensor != "03":
            out_indices.append(index)

        if args.debug:
            break

    out_df = df.iloc[out_indices]

    out_df.to_csv(args.output_csv, index=False)
