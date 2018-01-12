import argparse

import itertools
import random
import sys
import networkx as nx
import LinkPrediction
from collections import defaultdict

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

	print("Building graphs")
	X = [ nx.Graph() for _ in reltypesSeen ]
	for g in X:
		g.add_nodes_from(list(range(len(idsSeen))))

	for a,b,c in tuplesAsIDs:
		X[a].add_edge(b,c)

	perGraph = defaultdict(list)
	for (relID,eID1,eID2,isPos) in testingData:
		perGraph[relID].append((eID1,eID2))

	preds = {}
	for relID,edgesToPredict in perGraph.items():
		preds[relID] = LinkPrediction.CommonNeighbors(X[relID],edgesToPredict)

	print("Extracting score for test points")
	testingScores = []
	for (relID,eID1,eID2,isPos) in testingData:
		score = preds[relID][(eID1,eID2)]
		testingScores.append((score,isPos))

	print("Outputting to file")
	with open(args.outFile,'w') as outF:
		for score,isPos in testingScores:
			isPosBinary = 1 if isPos else 0
			outF.write("%f\t%d\n" % (score,isPosBinary))


