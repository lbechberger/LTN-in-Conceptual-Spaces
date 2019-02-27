# -*- coding: utf-8 -*-
"""
Implementation of the apriori algorithm for extracting frequent item sets.

Inspired by https://github.com/rasbt/mlxtend/blob/master/mlxtend/frequent_patterns/apriori.py.

Created on Tue Feb 26 10:56:28 2019

@author: lbechberger
"""

import argparse, os, pickle
import numpy as np

# parse command line arguments
parser = argparse.ArgumentParser(description='apriori algorithm for movie genres')                  
parser.add_argument('input_file', help = 'path to the file containing the data set information')
parser.add_argument('-o', '--output_folder', help = 'path to output folder for the pickle files', default = '.') 
parser.add_argument('-l', '--limit', type = int, help = 'maximal number of literals in rule', default = 2)
parser.add_argument('-s', '--support', type = float, help = 'minimal support of item sets', default = 0.008)               
parser.add_argument('-q', '--quiet', action="store_true", help = 'disables info output')                    
args = parser.parse_args()

# load data set
data_set = pickle.load(open(args.input_file, 'rb'))

positive_concepts = data_set['all_concepts']
positive_classifications = data_set['all_classifications']

negative_classifications = 1 - positive_classifications
negative_concepts = ['~{0}'.format(concept) for concept in positive_concepts]

all_concepts = positive_concepts + negative_concepts
all_classifications = np.concatenate([positive_classifications, negative_classifications], axis = 1)

# prepare data structure
frequent_itemsets = {}
supports = {}
size_of_data_set = all_classifications.shape[0]

# compute starting point manually
concept_frequencies = np.sum(all_classifications, axis = 0) / size_of_data_set
supports[1] = concept_frequencies[concept_frequencies >= args.support]
frequent_itemsets[1] = np.arange(len(all_concepts))[concept_frequencies >= args.support].reshape(-1, 1)

# helper function for generating candidate item sets
def generate_candidates(old_itemsets, last_iteration):
    individual_items = np.unique(old_itemsets.flatten())
    for old_itemset in old_itemsets:
        for individual_item in individual_items:
            # try to expand each of the old itemsets with each of the feasible concepts
            # the ordering constraint here only makes sure that no combination is generated twice
            if individual_item > max(old_itemset):
                new_candidate = tuple(old_itemset) + (individual_item,)
                # if we're in the last iteration: discard all sets with only negative literals (won't be used anyways)
                if last_iteration and min(new_candidate) >= len(positive_concepts):
                    continue
                yield new_candidate


# run apriori algorithm
for k in range(2, args.limit + 1):
    
    if not args.quiet:
        print("\tIteration", k)
        
    new_frequent_itemsets = []    
    new_supports = []

    # generate candidates
    candidates = generate_candidates(frequent_itemsets[k - 1], k == args.limit)   
    
    # filter candidates for minimal support
    for idx, candidate in enumerate(candidates):
        
        if not args.quiet and idx % (10**6) == 0:
            print("\t\tEvaluating candidate {0}".format(idx))
       
        # take all rows for which all specified columns are set to true
        support = all_classifications[:, candidate].all(axis = 1).sum() / size_of_data_set
        if support >= args.support:
            new_frequent_itemsets.append(candidate)
            new_supports.append(support)
    
    if len(new_frequent_itemsets) == 0:
        break
    
    # store results locally
    if not args.quiet:
        print("\t\tkept {0} itemsets".format(len(new_frequent_itemsets)))
    frequent_itemsets[k] = np.array(new_frequent_itemsets)
    supports[k] = np.array(new_supports)

# dump results into pickle file
central_output = {'itemsets': frequent_itemsets, 'supports': supports, 'concepts': all_concepts, 'border': len(positive_concepts)}
with open(os.path.join(args.output_folder, 'frequent_itemsets.pickle'), 'wb') as f:
    pickle.dump(central_output, f)
