# -*- coding: utf-8 -*-
"""
Splits the overall data set into training, validation, and test set.

Created on Wed Feb 13 07:56:25 2019

@author: lbechberger
"""

import argparse, os, pickle
import numpy as np
from sklearn.model_selection import train_test_split

# parse command line arguments
parser = argparse.ArgumentParser(description='split the data into training, validation, and test')                  
parser.add_argument('input_file', help = 'path to the input pickle file containing the overall data set')
parser.add_argument('output_folder', help = 'path to output folder for the data file')                
parser.add_argument('-t', '--test_size', type = float, help = 'percentage of data to use as test set', default = 0.2)
parser.add_argument('-v', '--validation_size', type = float, help = 'percentage of data to use as validation set', default = 0.2)
parser.add_argument('-s', '--seed', type = int, help = 'seed for random number generator', default = None)
parser.add_argument('-q', '--quiet', action="store_true", help = 'disables info output')                    
args = parser.parse_args()

# load data set
data_set = pickle.load(open(args.input_file, 'rb'))

#output = {'mds_space' : mds_vectors, 'projected_space' : projected_vectors, 
#          'keyword_labels' : keyword_labels, 'keyword_classifications' : keyword_classifications,
#          'genre_labels' : genre_labels, 'genre_classifications' : genre_classifications,
#          'rating_labels' : rating_labels, 'rating_classifications' : extended_rating_classifications,
#          'all_concepts' : all_concepts, 'all_classifications' : all_classifications}

# first split: testing and rest
first_split = train_test_split(data_set['mds_space'], data_set['projected_space'], 
                               data_set['keyword_classifications'], data_set['genre_classifications'], 
                               data_set['rating_classifications'], data_set['all_classifications'],
                               test_size = args.test_size, shuffle = True, random_state = args.seed)

# remainder, needs to be split again
X_mds = first_split[0]
X_projected = first_split[2]
y_keywords = first_split[4]
y_genres = first_split[6]
y_ratings = first_split[8]
y_all = first_split[10]

# create test set and store it
test_set = {'mds_space' : first_split[1], 'projected_space' : first_split[3], 
            'keyword_labels' : data_set['keyword_labels'], 'keyword_classifications' : first_split[5],
            'genre_labels' : data_set['genre_labels'], 'genre_classifications' : first_split[7],
            'rating_labels' : data_set['rating_labels'], 'rating_classifications' : first_split[9],
            'all_concepts' : data_set['all_concepts'], 'all_classifications' : first_split[11]}

with open(os.path.join(args.output_folder, 'test_set.pickle'), 'wb') as f:
    pickle.dump(test_set, f)

if not args.quiet:
    print('    Stored {0} data points as test set (corresponds to {1}% of the overall data set)'.format(len(first_split[1]), 100 * args.test_size))

# second split: training and validation
# recompute fraction to take from ramainder
validation_fraction = args.validation_size / (1 - args.test_size)
second_split = train_test_split(X_mds, X_projected, y_keywords, y_genres, y_ratings, y_all,
                                test_size = validation_fraction, shuffle = True, random_state = args.seed)

# training data, need to do additional balancing
X_mds_train = second_split[0]
X_projected_train = second_split[2]
y_keywords_train = second_split[4]
y_genres_train = second_split[6]
y_ratings_train = second_split[8]
y_all_train = second_split[10]

# create validation set and store it
validation_set = {'mds_space' : second_split[1], 'projected_space' : second_split[3], 
                  'keyword_labels' : data_set['keyword_labels'], 'keyword_classifications' : second_split[5],
                  'genre_labels' : data_set['genre_labels'], 'genre_classifications' : second_split[7],
                  'rating_labels' : data_set['rating_labels'], 'rating_classifications' : second_split[9],
                  'all_concepts' : data_set['all_concepts'], 'all_classifications' : second_split[11]}

with open(os.path.join(args.output_folder, 'validation_set.pickle'), 'wb') as f:
    pickle.dump(validation_set, f)

if not args.quiet:
    print('    Stored {0} data points as validation set (corresponds to {1}% of the overall data set)'.format(len(second_split[1]), 100 * args.validation_size))


# now create balanced versions of the training set (1/3-2/3 and 1/2-1/2)
occurrences = np.sum(y_all_train, axis = 0)
if not args.quiet:
    print('    Most infrequent class appers {0} times, most frequent class {1} times in data set of size {2}'.format(min(occurrences), max(occurrences), len(y_all_train)))
majority_classes =  np.where(occurrences > 0.5 * len(y_all_train), 1, 0)

# set random seed in order to ensure reproducibility
if args.seed != None:
    np.random.seed(args.seed)

# collect the indices for 33-67 balancing and for 50-50 balancing
third_indices_collection = []
half_indices_collection = []

# iterate over the individual concepts
for label_idx in range(len(data_set['all_concepts'])):
    # count majority and minority class
    majority_count = occurrences[label_idx] if majority_classes[label_idx] else len(y_all_train) - occurrences[label_idx]
    minority_count = len(y_all_train) - majority_count
    
    # get indices of majority and minority class
    majority_indices = np.nonzero(y_all_train[:,label_idx] == majority_classes[label_idx])[0]
    minority_indices = np.nonzero(y_all_train[:,label_idx] != majority_classes[label_idx])[0]    
    
    # use all indices by default
    third_indices_majority = majority_indices
    half_indices_majority = majority_indices
    
    # check for 33-67 balancing
    if majority_count > 2 * minority_count:
        third_indices_majority = np.random.choice(majority_indices, size = 2 * minority_count, replace = False)
        
    # check for 50-50 balancing
    if majority_count > minority_count:
        half_indices_majority = np.random.choice(majority_indices, size = minority_count, replace = False)
        
    # add the minority indices  
    third_indices = np.concatenate((third_indices_majority, minority_indices))
    half_indices = np.concatenate((half_indices_majority, minority_indices))
    
    # sort the whole thing
    third_indices = np.sort(third_indices)
    half_indices = np.sort(half_indices)
    
    # store in buffer
    third_indices_collection.append(third_indices)
    half_indices_collection.append(half_indices)


# create training set and store it
training_set = {'mds_space' : X_mds_train, 'projected_space' : X_projected_train, 
                'keyword_labels' : data_set['keyword_labels'], 'keyword_classifications' : y_keywords_train,
                'genre_labels' : data_set['genre_labels'], 'genre_classifications' : y_genres_train,
                'rating_labels' : data_set['rating_labels'], 'rating_classifications' : y_ratings_train,
                'all_concepts' : data_set['all_concepts'], 'all_classifications' : y_all_train,
                'balanced_third_indices': third_indices_collection, 'balanced_half_indices' : half_indices_collection}

with open(os.path.join(args.output_folder, 'training_set.pickle'), 'wb') as f:
    pickle.dump(training_set, f)

if not args.quiet:
    print('    Stored {0} data points as training set (corresponds to {1}% of the overall data set)'.format(len(X_mds_train), 100 * (1 - args.validation_size - args.test_size)))
