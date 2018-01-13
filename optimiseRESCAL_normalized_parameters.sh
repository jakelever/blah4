#!/bin/bash
#set -euxo pipefail

for rank in `seq 25 50`
do
	for alpha in 0.1 0.25 0.5 0.75 0.9
	do
		sh runRESCAL_normalized_parameters.sh $rank 0.0 0.0 $alpha
	done
done

