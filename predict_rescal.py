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
	parser = argparse.ArgumentParser()
	parser.add_argument('--trainingData')
	parser.add_argument('--validationData')
	parser.add_argument('--outFile')
	args = parser.parse_args()

	tuples = set()
	reltypesSeen = set()
	idsSeen = set()

	id2TypeTerm = {}

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

	index2id = sorted(list(idsSeen))
	index2reltype = sorted(list(reltypesSeen))

	ids2Index = { id:index for index,id in enumerate(index2id) }
	reltype2Index = { reltype:index for index,reltype in enumerate(index2reltype) }
	#reltype2Index['negative'] = -1

	tuplesAsIDs = [ (reltype2Index[reltype],ids2Index[id1],ids2Index[id2]) for reltype,id1,id2 in tuples ]

	validation = []
	with open(args.validationData) as f:
		for line in f:
			split = line.strip().split('\t')
			pmid,reltype,id1,type1,term1,id2,type2,term2,posOrNeg = split
			isPos = (posOrNeg == 'positive')

			tupleAsID = (reltype2Index[reltype],ids2Index[id1],ids2Index[id2],isPos)
			validation.append(tupleAsID)

	print("Building matrix time!")

	matrixSize = (len(reltypesSeen),len(idsSeen),len(idsSeen))
	X = [ lil_matrix((len(idsSeen),len(idsSeen))) for _ in reltypesSeen ]

	for a,b,c in tuplesAsIDs:
		#print(a,b,c)
		X[a][b,c] = 1

	#for _ in range(100000):
	#	a = random.randint(0,len(X)-1)
	#	b = random.randint(0,len(idsSeen)-1)
	#	c = random.randint(0,len(idsSeen)-1)
	#	X[a][b,c] = 1


	# Set logging to INFO to see RESCAL information
	logging.basicConfig(level=logging.INFO)

	# Load Matlab data and convert it to dense tensor format
	#T = loadmat('data/alyawarra.mat')['Rs']
	#X = [lil_matrix(T[:, :, k]) for k in range(T.shape[2])]

	# Decompose tensor using RESCAL-ALS
	#A, R, fit, itr, exectimes = rescal_als(X, 100, init='nvecs', lambda_A=10, lambda_R=10)
	preds = predict_rescal_als(X)
	#preds[preds < 0.1] = 0.0

	print(type(preds))
	print(preds.shape)

	idCount = len(idsSeen)
	reltypeCount = len(reltypesSeen)

	#preds = [ m.tocsr() for m in preds ]

	#for t in itertools.product(*preds.nonzero()):
	#	print(t)
	#	break
	#print(preds.nonzero())
	#assert False

	trainingScores = []
	for relID,eID1,eID2 in tuplesAsIDs:
		score = preds[eID1,eID2,relID]
		trainingScores.append((score,True))
	#print(trainingScores)

	validationScores = []
	for (relID,eID1,eID2,isPos) in validation:
		score = preds[eID1,eID2,relID]
		validationScores.append((score,isPos))

	#print(validationScores[:100])

	with open(args.outFile,'w') as outF:
		for score,isPos in validationScores:
			isPosBinary = 1 if isPos else 0
			outF.write("%f\t%d\n" % (score,isPosBinary))
	sys.exit(0)

	posCount = sum( 1 for _,isPos in validationScores if isPos )
	negCount = len(validationScores) - posCount

	reweight = posCount / float(preds.shape[0]*preds.shape[1]*preds.shape[2] - trainingSize)
	print('reweight',reweight)
	#assert False

	validationScores = sorted(validationScores,reverse=True)
	TP,FP = 0,0
	bestFScore = -1.0
	for _,isPos in validationScores:
		if isPos:
			TP += 1
		else:
			FP += 1

		TN = negCount - FP
		FN = posCount - TP

		precision,recall,fscore = 0,0,0
		if TP+FP != 0:
			precision = reweight*TP / float(reweight*TP + (1-reweight)*FP)
		if TP+FN != 0:
			recall = TP / float(TP+FN)
		if TP+FP != 0 and TP+FN != 0:
			fscore = 2 * (precision*recall) / (precision+recall)

		#print(TP,FP,TN,FN,precision,recall,fscore)
		if fscore > bestFScore:
			bestFScore = fscore
			print(TP,FP,TN,FN,precision,recall,fscore)
		

	#predsAsIDs = []
	#for i,j,k in zip(*preds.nonzero()):
	#	predsAsIDs

	sys.exit(0)
	with open(args.outFile,'w') as outF:
		#for i,j,k in itertools.product(range(idCount),range(idCount),range(reltypeCount)):
		for i,j,k in zip(*preds.nonzero()):
			#if X[k][i,j] == 1:
			if True:
				print(i,j,k)
				#print(preds[i,j,k])
				#break
				orig = X[k][i,j]
				score = preds[i,j,k]
				reltype,id1,id2 = index2reltype[k],index2id[j],index2id[i]
				type1,term1 = id2TypeTerm[id1]
				type2,term2 = id2TypeTerm[id2]

				outData = [orig,score,reltype,id1,type1,term1,id2,type2,term2]
				outLine = "\t".join(map(str,outData))
				#print(outLine)
				outF.write(outLine + "\n")
				#break
				
