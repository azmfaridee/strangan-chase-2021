# STranGAN

This repository provides PyTorch implementation of the following IEEE/ACM CHASE 2021 paper:

* [STranGAN: Adversarially-Learnt Spatial Transformer for Scalable Human Activity Recognition](https://doi.org/10.1016/j.smhl.2021.100226)

## Overview
<p align="center">
  <img src="https://raw.githubusercontent.com/azmfaridee/strangan-chase-2021/main/img/overview-strangan.png" width="800">
</p>

## Abstract

We tackle the problem of domain adaptation for inertial sensing-based human activity recognition (HAR) applications
-i.e., in developing mechanisms that allow a classifier trained on sensor samples collected under a certain narrow
context to continue to achieve high activity recognition accuracy even when applied to other contexts. This is a problem
of high practical importance as the current requirement of labeled training data for adapting such classifiers to every
new individual, device, or on-body location is a major roadblock to community-scale adoption of HAR-based applications.
We particularly investigate the possibility of ensuring robust classifier operation, without requiring any new labeled
training data, under changes to

1. the individual performing the activity, and
2. the on-body position where the sensor-embedded mobile or wearable device is placed.

We propose STranGAN, a framework that explicitly decouples the domain adaptation functionality from the classification
model by learning and applying a set of optimal spatial affine transformations on the target domain inertial sensor data
stream by employing adversarial learning, which only requires collecting raw data samples (but no accompanying activity
labels) from both source and target domains. STranGAN’s uniqueness lies in its ability to perform practically useful
adaptation

1. without any labeled training data and without requiring paired, synchronized generation of source and target domain
   samples, and
2. without requiring any changes to a pre-trained HAR classifier.

Empirical results using three publicly available benchmark datasets indicate that STranGAN

1. is particularly effective in handling on-body position heterogeneity (achieving a 5% improvement in classification F1
   score compared to state-of-the-art baselines),
2. offers competitive performance for handling cross-individual variations, and
3. the affine transformation parameters can be analyzed to gain interpretable insights on the domain heterogeneity.

## Installation

This repo was tested with Ubuntu 20.04, Python 3.8.10, PyTorch 1.9.0+cu111, and CUDA 11.2

1. Clone this repo with:
   ```shell
   git clone git@github.com:azmfaridee/strangan-chase-2021.git
   cd strangan-chase-2021.git
   ```

2. _Optional:_ you can consider setting up a docker environment. Here is a
   nice [set of scripts](https://github.com/mpsc-lab-umbc/docker-scripts/blob/master/README.md) on getting started with the
   same ML environment we used with docker.

3. Install packages:
    ```shell
    pip3 install -r requirements.txt
    ```

## Usage

Here is a simple example of how to use `STranGAN` to perform domain adaptation on OPPORTUNITY dataset.

```shell
PREFIX='/workspace/phd/strangan-chase-2021/src'
DATA_PATH='/workspace/phd/strangan/data/preprocessed/opportunity_all_users.npz'
SAVE_PREFIX='/tmp/strangan-runs/'
RUN_ID='2'
SAVE_DIR=$SAVE_PREFIX$RUN_ID

python $PREFIX/strangan.py -d $DATA_PATH \
  -ss 'S1,S2' \
  -st 'S3,S4' \
  -ps 'LUA' \
  -pt 'BACK' \
  -ch 3 \
  -cls 4 \
  -bs 32 \
  --n_epochs 3 \
  --gan_epochs 10 \
  --gpu 0 \
  --lr_FC 0.002 --lr_FC_b1 0.9 --lr_FC_b2 0.999 \
  --lr_FD 0.0002 \
  --lr_G 0.00002 --lr_G_b1 0.5 --lr_G_b2 0.999 \
  --gamma 0.9 \
  --save_dir $SAVE_DIR \
  --log_interval 50 \
  --eval_interval 500
```

You will need to pre-process the raw IMU data with imputation of missing values, filtering and sliding window based
segmentation. Please refer to the documentation of `ActivityDataset` in `src/dataset.py` for more details on the shape
of the input matrix.

## Citation

If you find this repository useful in your research, please consider citing our paper

```
@article{FARIDEE2021100226,
    title = {STranGAN: Adversarially-learnt Spatial Transformer for scalable human activity recognition},
    journal = {Smart Health},
    pages = {100226},
    year = {2021},
    issn = {2352-6483},
    doi = {https://doi.org/10.1016/j.smhl.2021.100226},
    url = {https://www.sciencedirect.com/science/article/pii/S2352648321000477},
    author = {Abu Zaher Md Faridee and Avijoy Chakma and Archan Misra and Nirmalya Roy},
    keywords = {Domain adaptation, Wearable sensing, Learnable data augmentation, Adversarial learning, Generative modeling},
}
```

## Contact

If you have any questions, please feel free to reach out over email via `faridee1@umbc.edu`

## Acknowledgements

This research is supported by NSF CAREER grant `1750936`, U.S. Army grant `W911NF2120076`, ONR grant `N00014-18-1-2462`, and
Alzheimer’s Association grant `AARG-17-533039`.
