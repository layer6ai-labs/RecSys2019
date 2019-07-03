<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>

# 2019 ACM RecSys Challenge 2'nd Place Solution

## Introduction



## Environment

The model is implemented in Java and tested on the following environment:

* Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz
* 256GB RAM
* Nvidia Titan V
* Java Oracle 1.8.0_171
* Apache Maven 3.3.9
* Intel MKL 2018.1.038
* XGBoost and XGBoost4j 0.90

## Execution

Please use our `run.sh` script provided for end-to-end compilation and execution of our entire pipeline (data parsing, feature extraction, training, validation, and submission) by following these steps:

1) Set your data directory `dataPath` to point to the directory where you downloaded the official [RecSys 2019 Trivago dataset (Version 2)](https://recsys.trivago.cloud/challenge/dataset/) to. Verify that your `dataPath` contains `train.csv`, `test.csv`, and `item_metadata.csv` first!

2) Set your output directory `outPath` to point to the directory where our code will output all relevant files to.

3) Set the model version `modelVersion` to either be `1` or `2` based on our two provided sets of XGB training hyperparameters. Model version `1` trains XGB using a histogram tree method approach with minimal regularization and will achieve `AUC_valid ~ 0.9238, MRR_valid ~ 0.6747` in a runtime of ~1 hours. Model version `2` trains XGB using an exact tree method approach with heavy regularization and will achieve a higher score of `AUC_valid ~ , MRR_valid ~ ` at the cost of a much longer runtime of ~2-3 days.

4) Execute `./run.sh`

Once the above run is finished, you can locate the final submission file `submit.csv` in your specified output path `outPath`. We prioritized speed over memory for this project so please use a machine with at least 200GB of RAM to run our model training and inference.

## Results

#### Data Parsing / Feature Extraction

We built our training and validation instances by treating each impression item in each session as an individual instance.
Each instance has a binary target label determining whether the customer actually clicked out on the impression item, and a feature vector of length 330 which includes
```
Impressions item features:
  * Item appearance rank
  * Item appearance rank within star group
  * Item appearance rank within rating group
  * Item price rank
  * Item price rank within star group
  * Item price rank within rating group
  * Item price rank within groups of higher appearance rank
  * Item price
  * Item price and median price difference
  * Item metadata properties
  * Global user action count, appearance rank, price rank averages
  * Global item action count, appearance rank, price rank, star/rating group information
  * Global item-item user action interaction scores
  * Global user-user user action interaction scores
  * Global item-item user impression interaction scores
  * Global user-user user impression interaction scores
  * Local and global item price count differences
  * Local and global item action count and rank differences
  * Local and global user action count and rank differences
  
Impression item summarization features:
  * Mean appearance prices across top _k_ appearance rank items
  * Mean appearance price ranks across top _k_ appearance rank items
  * Mean of impression properties across impressions
  * Mean of global item counts across impressions
  * Entropy of counts
  * Entropy of properties
  
Session features: 
  * Stats on last 2 item action interactions prior to clickout
  * Stats on last action interactions prior to clickout
  * Impression length at clickout
  * Step number at clickout
  * Time duration between clickout and session start
  * Device
  
Non-item features:
  * Global rank counts
  * Global price rank counts
  * Global platform counts
  * Global city counts
  * Global device counts
```

Given that there are ~900k total number of sessions along with the fact that most of them consists of 25 impression items, we are left with under 22M instances available for us to train and validate a model.
We further downsample to negative samples to 20 negative samples per positive sample to arrive at a training set with 14.5M instances, and a validation set of 1.8M instances to train an XGB model.

#### Training an XGB model

The XGB training hyper parameters we use for training model versions 1 and 2 are
```
* booster = gbtree
* eta = 0.1
* gamma = 0
* min_child_weight = 1
* max_depth = 10
* subsample = 1
* colsample_bynode = 0.8
* scale_pos_weight = 1
* objective = binary:logistic
* base_score = 0.1
* seed = 3
* lambda = 1 [version 1] or 4000 [version 2]
* alpha = 0 [version 1] or 10 [version 2]
* tree_method = hist [version 1] or exact [version 2]
```

We found that that the AUC and MRR evaluation metrics were closely correlated in this competition, and for convenience maximized the validation AUC during training. By re-running the code provided in this repository, we reproduce results of


| Model version | # of features | Iterations | Runtime (hours) | AUC (valid) | MRR (valid) | MRR (test) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 330 | 435 | 1 | 0.9238 | 0.6747 | ~0.683 |
| 2 | 330 |   | 60 | 0.9258 | 0.6774 | ~0.685 |


Our true final competition submission achieves `MRR (test) ~ 0.688` via a 2nd-stage blending of multiple XGB, RNN, and Transformer models which we detail in our corresponding workshop paper.