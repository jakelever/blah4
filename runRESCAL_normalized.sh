#!/bin/bash
set -euxo pipefail

rank=$1
lambda_A=$2
lambda_R=$3

name="$rank\_$lambda_A\_$lambda_R"

mkdir -p intermediate
mkdir -p crossvalidated

python predict_rescal_normalized.py --trainingEdgeWeights data/Chemical_Disease.tsv,data/Chemical_Gene.tsv,data/Disease_Gene.tsv --trainingEdgeTypes medical_condition_treated,physically_interacts,genetic_association --testingData data/test.all.tsv --rank $rank --lambda_A $lambda_A --lambda_R $lambda_R --outFile intermediate/rescalnorm_$name
python evaluate.py --data intermediate/rescalnorm_$name --classbalance 0.000131542 > crossvalidated/rescalnorm_$name


