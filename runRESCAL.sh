#!/bin/bash

rank=$1
lambda_A=$2
lambda_R=$3

name="$rank\_$lambda_A\_$lambda_R"

mkdir -p intermediate
mkdir -p crossvalidated

python predict_rescal.py --trainingData data/validation.train.tsv --testingData data/validation.test.tsv --outFile intermediate/rescal_$name --rank $rank --lambda_A $lambda_A --lambda_R $lambda_R
python evaluate.py --data intermediate/rescal_$name --classbalance 0.000131542 > crossvalidated/out_$name


