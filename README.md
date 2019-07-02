# 2019 ACM RecSys Challenge 2'nd Place Solution by Layer6 AI

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
