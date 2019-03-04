#!/bin/bash
set -euxo pipefail

rank=30
lambda_A=0
lambda_R=0
alpha=0.9

classbalance=0.000106543
#classbalance=0.000980235

files=data/moreData/Chemical_Disease_cutoffyear_2010_weightingexponent_$alpha.tsv,data/moreData/Chemical_Gene_cutoffyear_2010_weightingexponent_$alpha.tsv,data/moreData/Disease_Gene_cutoffyear_2010_weightingexponent_$alpha.tsv
python predict_rescal_normalized.py --trainingEdgeWeights $files --trainingEdgeTypes medical_condition_treated,physically_interacts,genetic_association --testingData data/test.all.tsv --rank $rank --lambda_A $lambda_A --lambda_R $lambda_R --outFile tmp2
python evaluate.py --data tmp2 --classbalance $classbalance --prCurveData prCurves/rescalnormalpha.tsv > results/rescalnormalpha.txt

