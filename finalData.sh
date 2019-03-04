#!/bin/bash

classbalance=0.000106543
#classbalance=0.000980235

python predict_rescal.py --trainingData data/train.all.tsv --testingData data/test.all.tsv --rank 7 --lambda_A 0.1 --lambda_R 0.5 --outFile tmp
python evaluate.py --data tmp --classbalance $classbalance --prCurveData prCurves/rescal.tsv > results/rescal.txt

for method in DegreeProduct CommonNeighbors Jaccard Sorensen LHN ShortestPath ResourceAllocation AdamicAdvar
do
	python predict_classiclinkprediction.py --trainingData data/train.all.tsv --testingData data/test.all.tsv --method $method --outFile tmp
	python evaluate.py --data tmp --classbalance $classbalance --prCurveData prCurves/$method.tsv > results/$method.txt
done

