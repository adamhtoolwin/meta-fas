# Meta-learning for efficient few-shot classification in Facial Liveness Detection

## Prerequisites

1. Run `pip install requirements.txt` to install all the required packages.

## How to use

1. Run all training and validation files with the `-c` argument and give the path to the config file. 
Some sample config files are given in the [configs folder](/configs) and can be customized.
For e.g. To run a 5-shot OULU meta-training experiment, `python meta-oulu.py -c configs/oulu/5-shot/train.config`.

**Note**: All files when run for the first time will take some time to generate the metadataset based on the actual data.
This metadataset should be saved in the file specified in the `bookkeeping_path` in the configs files and later runs will reuse this file.

## Examples

1. MNIST examples under the mnist folder. A sample config file for it is also given and can be customized.