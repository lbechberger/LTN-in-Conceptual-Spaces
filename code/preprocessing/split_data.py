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
parser.add_argument('-a', '--analyze', action="store_true", help = 'display analysis of the three resulting data sets')
parser.add_argument('-q', '--quiet', action="store_true", help = 'disables info output')                    
args = parser.parse_args()

# load data set
data_set = pickle.load(open(args.input_file, 'rb'))

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

# do an extra analysis
if args.analyze and not args.quiet:
    
    def distinct_label_set(classifications):
        """Computes the distinct label set of the given data set.
        
        How many distinct label combinations are there in the data set?"""
    
        label_sets = []   
        for line in classifications:
            label_sets.append(str(line))
        
        return len(set(label_sets))
    
    def proportion_of_distinct_label_set(classifications):
        """Computes the proportion of distinct label set of the given data set.
        
        Distinct label set normalized by the total number of training instances."""
        
        return (1.0 * distinct_label_set(classifications)) / classifications.shape[0]
    
    def label_cardinality(classifications):
        """Computes the label cardinality of the given data set.
        
        How many labels per example do we have on average?"""
        
        return np.mean(np.sum(classifications, axis = 1))        
        
    
    def label_density(classifications):
        """Computes the label density of the given data set.
        
        Label cardinality normalized by the total number of labels."""
        
        return (1.0 * label_cardinality(classifications)) / classifications.shape[1]
    
    def label_distribution(classifications):
        """Computes the distribution of labels for the given data set.
        
        How often do the labels occur percentage-wise in the data set?"""
        
        counts = np.sum(classifications, axis = 0)
        frequencies = counts / classifications.shape[0]        
        
        return frequencies

    def analyze_subset(subset):
        """Analyzes the given data subset."""
        
        y_all = subset['all_classifications']
        y_keywords = subset['keyword_classifications']
        y_genres = subset['genre_classifications']
        y_ratings = subset['rating_classifications']
        print('\t\tDistinct label set: {0} (keywords: {1}, genres: {2}, ratings: {3})'.format(distinct_label_set(y_all), 
                          distinct_label_set(y_keywords), distinct_label_set(y_genres), distinct_label_set(y_ratings)))
        print('\t\tProportion of distinct label set: {0} (keywords: {1}, genres: {2}, ratings: {3})'.format(proportion_of_distinct_label_set(y_all), 
                          proportion_of_distinct_label_set(y_keywords), proportion_of_distinct_label_set(y_genres), proportion_of_distinct_label_set(y_ratings)))
        print('\t\tLabel cardinality: {0} (keywords: {1}, genres: {2}, ratings: {3})'.format(label_cardinality(y_all), 
                          label_cardinality(y_keywords), label_cardinality(y_genres), label_cardinality(y_ratings)))
        print('\t\tLabel density: {0} (keywords: {1}, genres: {2}, ratings: {3})'.format(label_density(y_all), 
                          label_density(y_keywords), label_density(y_genres), label_density(y_ratings)))
        return label_distribution(y_all)
    
    print('\tAnalyzing training set')
    frequencies_training = analyze_subset(training_set)
    print('\tAnalyzing validation set')
    frequencies_validation = analyze_subset(validation_set)
    print('\tAnalyzing test set')
    frequencies_test = analyze_subset(test_set)
    
    print('\tLabel frequencies:')
    for label, freq_train, freq_valid, freq_test in zip(data_set['all_concepts'], frequencies_training, frequencies_validation, frequencies_test):
        print('\t\t{0} \t- train: {1} - validation: {2} - test: {3}'.format(label, freq_train, freq_valid, freq_test))
    
    from matplotlib import pyplot as plt
    for frequencies, set_name in [(frequencies_training, 'training'), (frequencies_validation, 'validation'), (frequencies_test, 'test')]:   
        plt.hist(frequencies)
        plt.title('distribution of label frequencies in {0} set'.format(set_name))
        output_file_name = os.path.join(args.output_folder, '{0}.png'.format(set_name))
        plt.savefig(output_file_name)
        plt.close()
                          