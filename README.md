# Meta-learning for efficient few-shot classification in Facial Liveness Detection

## Prerequisites

1. Run `pip install requirements.txt` to install all the required packages.

## How to use

1. Run all training and validation files with the `-c` argument and give the path to the config file. 
Some sample config files are given in the [configs folder](/configs) and can be customized.
For e.g. To run a 5-shot OULU meta-training experiment, `python meta-oulu.py -c configs/oulu/5-shot/train.config`.

**Note**: All files when run for the first time will take some time to generate the metadataset based on the actual data.
This metadataset should be saved in the file specified in the `bookkeeping_path` in the configs files and later runs will reuse this file.
I think this depends on the seed given in the configs otherwise learn2learn will index the files differently each run so the seed should be set in the configs.

## Examples

1. MNIST examples under the [mnist](/examples/mnist) folder. A sample config file for it is also given and can be customized.
Run `python meta_mnist.py -c configs/mnist/train_config.yml`.

## Dataset Preprocessing

A script to decode HKBU-MARs is provided in [preprocessing](/preprocessing) folder.

## References

1. Finn, C., Abbeel, P., & Levine, S. (2017, July). Model-agnostic meta-learning for fast adaptation of deep networks. 
In International Conference on Machine Learning (pp. 1126-1135). PMLR.

2. Feng, H., Hong, Z., Yue, H., Chen, Y., Wang, K., Han, J., ... & Ding, E. (2020). 
Learning generalized spoof cues for face anti-spoofing. arXiv preprint arXiv:2005.03922.
