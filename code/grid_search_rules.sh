#!/bin/bash

for support in 0.01 0.02 0.05 0.1
do
	for confidence in 0.80 0.85 0.90 0.95
	do
		folder='data/Ager/rules_exploration/s'"$support"'_c'"$confidence"
		mkdir -p $folder
		python code/preprocessing/extract_rules.py data/Ager/preprocessed/counts/central.pickle -s $support -c $confidence -o $folder -q >> data/Ager/rules_exploration/summary.txt
	done
done
