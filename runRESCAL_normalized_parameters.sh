#!/bin/bash
set -euxo pipefail

rank=$1
lambda_A=$2
lambda_R=$3
alpha=$4

name=$rank"_"$lambda_A"_"$lambda_R"_"$alpha

mkdir -p intermediate
mkdir -p crossvalidated

files=data/moreData/Chemical_Disease_cutoffyear_2005_weightingexponent_$alpha.tsv,data/moreData/Chemical_Gene_cutoffyear_2005_weightingexponent_$alpha.tsv,data/moreData/Disease_Gene_cutoffyear_2005_weightingexponent_$alpha.tsv
python predict_rescal_normalized.py --trainingEdgeWeights $files --trainingEdgeTypes medical_condition_treated,physically_interacts,genetic_association --testingData data/validation.test.tsv --rank $rank --lambda_A $lambda_A --lambda_R $lambda_R --outFile intermediate/rescalnormalpha_$name
python evaluate.py --data intermediate/rescalnormalpha_$name --classbalance 0.000131542 > crossvalidated/rescalnormalpha_$name


