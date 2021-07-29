### Preprocessing

This folder contains various preprocessing scripts for frame extraction for the experiment datasets. 
Pytorch Datasets are found in datasets folder.

***

### Meta-training

1. [OULU-NPU](https://sites.google.com/site/oulunpudatabase/)
2. [CelebA-Spoof](https://github.com/Davidzhangyuanhan/CelebA-Spoof)

#### Protocol 1

- Train: OULU(live and print) + CelebA_Spoof
- Validation: OULU(replay)

#### Protocol 2

- Train: OULU(live and replay) + CelebA_Spoof
- Validation: OULU(print)

***

### Meta-testing

1. [HKBU - MARs V2](http://rds.comp.hkbu.edu.hk/mars/)
