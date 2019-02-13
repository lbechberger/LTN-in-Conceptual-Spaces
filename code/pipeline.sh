#!/bin/bash

# create necessary directories
mkdir -p data/Ager/preprocessed data/Ager/preprocessed/counts data/Ager/preprocessed/rules-strict data/Ager/preprocessed/rules-lean

# PREPROCESSING
echo 'PREPROCESSING'

# merge all individual files into a global one
echo 'merge files'
python code/preprocessing/merge_all_data.py data/Ager/raw_files/num_stw_100_MDS.npy data/Ager/raw_files/ranks.txt data/Ager/raw_files/keywords/ data/Ager/raw_files/genres/ data/Ager/raw_files/ratings/ data/Ager/preprocessed/

# split into training, validation, and test set
echo 'split data'
python code/preprocessing/split_data.py data/Ager/preprocessed/full_data_set.pickle data/Ager/preprocessed -t 0.2 -v 0.2 -s 42 -a

# do rule counting
if [ ! -f data/Ager/preprocessed/counts/central.pickle ]; then
    	echo 'counting co-occurences for rule statistics'
	python code/preprocessing/count_occurrences.py data/Ager/preprocessed/training_set.pickle -o data/Ager/preprocessed/counts
else
	echo 'count files already exist; using existing files'
fi

echo 'extracting strict rules'
python code/preprocessing/extract_rules.py data/Ager/preprocessed/counts/central.pickle -s 0.01 -i 0.1 -c 0.9 -o data/Ager/preprocessed/rules-strict
echo 'extracting lean rules'
python code/preprocessing/extract_rules.py data/Ager/preprocessed/counts/central.pickle -s 0.005 -i 0.05 -c 0.8 -o data/Ager/preprocessed/rules-lean
