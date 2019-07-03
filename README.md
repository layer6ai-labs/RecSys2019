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

1) Set your data directory `dataPath` to where you downloaded the official [RecSys 2019 Trivago dataset (V2)](https://recsys.trivago.cloud/challenge/dataset/) to. Verify that your `dataPath` contains `train.csv`, `test.csv`, and `item_metadata.csv` first!

2) Set your output directory `outPath` to where our code will output all relevant files to.

3) Set the model version `modelVersion` to `1` or `2` based on our two provided sets of XGB training hyper parameters. Model version `1` achieves `MRR_valid ~ 0.6747` over a 1 hour runtime. Model version `2` achieves a higher `MRR_valid ~ 0.6775` over ~1.5 days runtime. Please refer to the results section below for more details.

4) Execute `./run.sh`

Once the above run is finished, you can locate the final submission file `submit.csv` in your specified output directory `outPath`. We prioritized speed over memory for this project so please use a machine with at least 200GB of RAM to run our model training and inference.

## Results

#### Data Parsing / Feature Extraction

We built our training and validation instances by treating each impression item of each session as an individual instance.
Each instance is assigned a binary target label of whether the impression item was clicked during clickout, as well as a feature vector of length 330 containing quantities such as
```
Impression Item Features
  * Item appearance rank
  * Item appearance rank within same star group
  * Item appearance rank within same rating group
  * Item price rank
  * Item price rank within same star group
  * Item price rank within same rating group
  * Item price rank within groups of higher appearance rank
  * Item price
  * Item price and median price difference
  * Item metadata properties
  * Global item action count, appearance rank, price rank, star/rating group information
  * Global item-item user action interaction scores
  * Global user-user user action interaction scores
  * Global item-item user impression interaction scores
  * Global user-user user impression interaction scores
  * Local and global item price count differences
  * Local and global item action count and rank differences
  * Local and global user action count and rank differences
  
Summarization-Over-Impression-Items Features
  * Mean appearance prices across top _k_ appearance rank items
  * Mean appearance price ranks across top _k_ appearance rank items
  * Mean of impression properties across impressions
  * Mean of global item counts across impressions
  * Entropy of counts
  * Entropy of properties
  
Session Features
  * Stats on last 2 item action interactions prior to clickout
  * Stats on last action interactions prior to clickout
  * Impression length at clickout
  * Step number at clickout
  * Device at clickout
  * Time duration between clickout and session start
  
Non-Item Features
  * Global user counts
  * Global rank counts
  * Global price rank counts
  * Global platform counts
  * Global city counts
  * Global device counts
```

Given that there are ~900k total number of sessions in the dataset along with most of them consisting of 25 impression items each, we had under 22M total instances available for us to train and validate a model.
By further downsampling our negative samples to 20 negative samples per positive sample, and partitioning out ~11% of the total instances for validation, we arrived at a final training set of 14.5M instances and a validation set of 1.8M instances.

#### Training an XGB model

The XGB training hyper parameters we provide for training model versions `1` and `2` are
```
booster = gbtree
eta = 0.1
gamma = 0
min_child_weight = 1
max_depth = 10
subsample = 1
colsample_bynode = 0.8
scale_pos_weight = 1
bjective = binary:logistic
base_score = 0.1
seed = 3
lambda = 1 [version 1] or 4000 [version 2]
alpha = 0 [version 1] or 10 [version 2]
tree_method = hist [version 1] or exact [version 2]
```

Model version `1` trains an XGB model quickly via a histogram tree method with almost no regularization. Model version `2` trains a more accurate and generalizable XGB model via an exact tree method with heavy regularization.

For practical purposes, we found the AUC and MRR evaluation metrics to be closely correlated and for this reason maximized the validation AUC during training with early stopping. By running the code provided in this repository, we reproduced results of

| XGB model version | # of features | Early stopping rounds | Rounds | Runtime (hours) | AUC (valid) | MRR (valid) | MRR (test) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 330 | 10 | 435 | 1 | 0.9238 | 0.6747 | ~0.683 |
| 2 | 330 | 20 | 2820 | 32 | 0.9254 | 0.6775 | ~0.685 |

Our final competition submission achieves `MRR (test) ~ 0.688` via a 2nd-stage blending of multiple XGB, RNN, and Transformer models which we detail in our corresponding workshop paper.