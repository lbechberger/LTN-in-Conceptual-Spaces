#!/bin/bash

# create necessary directories
mkdir -p data/Ager/preprocessed

# preprocess the data: merge all individual files into a global one
python code/preprocessing/merge_all_data.py data/Ager/raw_files/num_stw_100_MDS.npy data/Ager/raw_files/ranks.txt data/Ager/raw_files/keywords/ data/Ager/raw_files/genres/ data/Ager/raw_files/ratings/ data/Ager/preprocessed/
