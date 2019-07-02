# 2019 ACM RecSys Challenge 2'nd Place Team Layer 6 AI

<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>

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

Once the above run is finished, you can locate the final submission file `submit.csv` in your specified output directory `outPath`. We prioritized speed over memory for this project so please use a machine with at least 200GB of RAM to run our model training and inference.

## Results



#### 1) Data Parsing

`train.csv`
nItems = 927940
sessionCount = 910732

nValidSessions = 78606

`test.csv`
nItems = 927997
sessionCount = 291381


#### 2) Feature Extraction

We built our training and validation instances for each of our at most 25 impression item in each of our 910732 sessions.
Each instance contains 330 features. and a binary target to indicate whether a user did end up clicking out on the impression item at clickout. 
As there are typically 25 impression items at clickout, we naturally have a positive:negative ratio of 1:25, which we further downsample to 1:20 for training.
Finally, we end with 14.5M instances for training and 1.8M instances for validation, in which we optimize for the validation AUC after verifying that the AUC and MRR are closely correlated in this competition.

The type of features we extract that make up for our 330 features include:

* User features: action counts, appearance rank counts, price rank counts

* Item features: 

* User-item features:

* Session features:

* Appearance rank, price rank, 

* Mean aggregation of prices and price ranks of high appearance rank impressions

* Price differences between impression item 

* Counts based on impression appearance rank, price rank, platform, city, and device

* One-hot encoding of item properties, and device

#### 3) Training

The shared XGB training hyperparameters we use for training model versions 1 and 2 are:
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
* lambda = 1 (version 1) / 4000 (version 2)
* alpha = 0 (version 1) / 10 (version 2)
* tree_method = hist (version 1) / exact (version 2) 
```

From running our code provided, we reproduce results of

| Model version | Number of features | Lambda (L2) | Alpha (L1) | Tree method | Training iterations | Training time (hours) | AUC (valid) | MRR (valid) | MRR (test/leaderboard) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 330 | 1 | 0 | hist |  435 | 1 | 0.9238 | 0.6747 | ~0.683 |
| 2 | 330 | 4000 | 10 | exact |  | 60 | 0.9258 | 0.6774 | ~0.685 |