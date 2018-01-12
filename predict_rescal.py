import argparse

import logging
from scipy.io.matlab import loadmat
from scipy.sparse import lil_matrix
from rescal import rescal_als
from numpy import dot,zeros
import itertools
import random
import sys

def predict_rescal_als(T):
	A, R, _, _, _ = rescal_als(
		T, 10, init='nvecs', conv=1e-4,
		lambda_A=0, lambda_R=0
	)
	n = A.shape[0]
	P = zeros((n, n, len(R)))
	for k in range(len(R)):
		print(A.shape, R[k].shape)
		print(R[k])
		P[:, :, k] = dot(A, dot(R[k], A.T))
	return P

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Make link predictions using the RESCAL ALS algorithm')
	parser.add_argument('--trainingData',required=True,type=str,help='Training data with tab-delimited columns: pmid,reltype,id1,type1,term1,id2,type2,term2')
	parser.add_argument('--testingData',required=True,type=str,help='Testing data with tab-delimited columns: pmid,reltype,id1,type1,term1,id2,type2,term2,posOrNeg')
	parser.add_argument('--outFile',required=True,type=str,help='Output file (tab-delimited) of score and pos/neg as 1/0')
	args = parser.parse_args()

	tuples = set()
	reltypesSeen = set()
	idsSeen = set()

	id2TypeTerm = {}

	print("Loading training data")
	with open(args.trainingData) as f:
		for line in f:
			split = line.strip().split('\t')
			pmid,reltype,id1,type1,term1,id2,type2,term2 = split

			tuples.add((reltype,id1,id2))
			idsSeen.add(id1)
			idsSeen.add(id2)
			reltypesSeen.add(reltype)

			id2TypeTerm[id1] = (type1,term1)
			id2TypeTerm[id2] = (type2,term2)

	trainingSize = len(tuples)

	print("Identifying indices and constructing data for sparse matrix")
	index2id = sorted(list(idsSeen))
	index2reltype = sorted(list(reltypesSeen))

	ids2Index = { id:index for index,id in enumerate(index2id) }
	reltype2Index = { reltype:index for index,reltype in enumerate(index2reltype) }

	tuplesAsIDs = [ (reltype2Index[reltype],ids2Index[id1],ids2Index[id2]) for reltype,id1,id2 in tuples ]

	print("Loading testing data")
	testingData = []
	with open(args.testingData) as f:
		for line in f:
			split = line.strip().split('\t')
			pmid,reltype,id1,type1,term1,id2,type2,term2,posOrNeg = split
			isPos = (posOrNeg == 'positive')

			tupleAsID = (reltype2Index[reltype],ids2Index[id1],ids2Index[id2],isPos)
			testingData.append(tupleAsID)

	print("Building sparse matrix")
	matrixSize = (len(reltypesSeen),len(idsSeen),len(idsSeen))
	X = [ lil_matrix((len(idsSeen),len(idsSeen))) for _ in reltypesSeen ]
	for a,b,c in tuplesAsIDs:
		X[a][b,c] = 1

	print("Running RESCAL")
	logging.basicConfig(level=logging.INFO)
	preds = predict_rescal_als(X)

	print("Extracting score for test points")
	idCount = len(idsSeen)
	reltypeCount = len(reltypesSeen)

	testingScores = []
	for (relID,eID1,eID2,isPos) in testingData:
		score = preds[eID1,eID2,relID]
		testingScores.append((score,isPos))

	print("Outputting to file")
	with open(args.outFile,'w') as outF:
		for score,isPos in testingScores:
			isPosBinary = 1 if isPos else 0
			outF.write("%f\t%d\n" % (score,isPosBinary))
