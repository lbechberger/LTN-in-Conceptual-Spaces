#!/bin/bash

# create necessary directories
mkdir -p data/Ager/preprocessed

# PREPROCESSING
echo 'PREPROCESSING'

# merge all individual files into a global one
echo 'merge files'
python code/preprocessing/merge_all_data.py data/Ager/raw_files/num_stw_100_MDS.npy data/Ager/raw_files/ranks.txt data/Ager/raw_files/keywords/ data/Ager/raw_files/genres/ data/Ager/raw_files/ratings/ data/Ager/preprocessed/

# split into training, validation, and test set
echo 'split data'
python code/preprocessing/split_data.py data/Ager/preprocessed/full_data_set.pickle data/Ager/preprocessed -t 0.2 -v 0.2 -s 42
