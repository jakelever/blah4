#!/bin/bash
#set -euxo pipefail

for rank in `seq 1 30`
do
	for lambda_A in `seq 0 9`
	do
		for lambda_R in `seq 0 9`
		do
#			python predict_rescal.py --trainingData validation.train.tsv --testingData validation.test.tsv --outFile tmp --rank $rank --lambda_A 0.$lambda_A --lambda_R 0.$lambda_R
#python evaluate.py --data tmp --classbalance 0.000131542 > crossvalidated/out_$rank\_0.$lambda_A\_0.$lambda_R

			echo "sh runRESCAL.sh $rank 0.$lambda_A 0.$lambda_R"
		done
	done
done

