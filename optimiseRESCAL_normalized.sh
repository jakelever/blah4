#!/bin/bash
#set -euxo pipefail

for rank in `seq 1 100`
do
	sh runRESCAL_normalized.sh $rank 0.0 0.0
done

