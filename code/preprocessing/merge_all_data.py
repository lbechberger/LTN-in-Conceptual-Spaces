# -*- coding: utf-8 -*-
"""
Merges all originally individual files into a single coherent data structure.

Created on Tue Feb 12 16:19:32 2019

@author: lbechberger
"""

import argparse, os, pickle
import numpy as np

# parse command line arguments
parser = argparse.ArgumentParser(description='merge all data into a single data structure')                  
parser.add_argument('space_file', help = 'path to the file containing the MDS space')
parser.add_argument('projection_file', help = 'path to the file containing the projected coordinates')
parser.add_argument('keywords_folder', help = 'path to the folder containing label information about plot keywords')
parser.add_argument('genres_folder', help = 'path to the folder containing label information about genres')
parser.add_argument('ratings_folder', help = 'path to the folder containing label information about ratings')
parser.add_argument('output_folder', help = 'path to output folder for the data file')                
parser.add_argument('-q', '--quiet', action="store_true", help = 'disables info output')                    
args = parser.parse_args()

# load MDS space
mds_vectors = np.load(args.space_file)

# load projected space
projected_vectors = np.loadtxt(args.projection_file).T # need to transpose the result

# load plot keyword data
keyword_labels = []
with open(os.path.join(args.keywords_folder, 'names.txt'), 'r') as f:
    for line in f:
        keyword_labels.append(line.replace('\n', '').replace('class-', ''))
keyword_classifications = np.load(os.path.join(args.keywords_folder, 'class-all.npy'))

# load genre data
genre_labels = []
with open(os.path.join(args.genres_folder, 'names.txt'), 'r') as f:
    for line in f:
        genre_labels.append(line.replace('\n', ''))
genre_classifications = np.load(os.path.join(args.genres_folder, 'class-all.npy'))

# load rating data
rating_labels = []
with open(os.path.join(args.ratings_folder, 'names.txt'), 'r') as f:
    for line in f:
        rating_labels.append(line.replace('\n', ''))
rating_classifications = np.load(os.path.join(args.ratings_folder, 'class-all.npy'))

# extend rating data such that we have a matrix of equal size
extended_rating_classifications = np.zeros((mds_vectors.shape[0], len(rating_labels)), dtype = int)

rating_map = np.loadtxt(os.path.join(args.ratings_folder, 'matched_ids.txt'))
rating_map = [int(x) for x in rating_map]

for rating_idx, item_idx in enumerate(rating_map):
    extended_rating_classifications[item_idx] = rating_classifications[rating_idx]

all_concepts = keyword_labels + genre_labels + rating_labels
all_classifications = np.concatenate((keyword_classifications, genre_classifications, extended_rating_classifications), axis = 1)
    
output = {'mds_space' : mds_vectors, 'projected_space' : projected_vectors, 
          'keyword_labels' : keyword_labels, 'keyword_classifications' : keyword_classifications,
          'genre_labels' : genre_labels, 'genre_classifications' : genre_classifications,
          'rating_labels' : rating_labels, 'rating_classifications' : extended_rating_classifications,
          'all_concepts' : all_concepts, 'all_classifications' : all_classifications}

with open(os.path.join(args.output_folder, 'full_data_set.pickle'), 'wb') as f:
    pickle.dump(output, f)    
    
    