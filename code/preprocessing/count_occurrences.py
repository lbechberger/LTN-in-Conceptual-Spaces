# -*- coding: utf-8 -*-
"""
Count occurrences of label combinations in order to estimate confidence of rules.

Created on Tue Feb 12 14:53:09 2019

@author: lbechberger
"""

import argparse, os, pickle
import numpy as np

# parse command line arguments
parser = argparse.ArgumentParser(description='rule extraction by label counting')                  
parser.add_argument('keywords_folder', help = 'path to the folder containing label information about plot keywords')
parser.add_argument('genres_folder', help = 'path to the folder containing label information about genres')
parser.add_argument('-o', '--output_folder', help = 'path to output folder for the pickle files', default = 'output/counts/')                
parser.add_argument('-q', '--quiet', action="store_true", help = 'disables info output')                    
args = parser.parse_args()

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

# merge them
all_concepts = keyword_labels + genre_labels
all_classifications = np.concatenate((keyword_classifications, genre_classifications), axis = 1)

# rule types we're interested in. "p" means "positive", "n" means "negative", 
#"IMPL" means "implies", and "DIFF" means "is different from"
rule_types = ['pIMPLp', 'pIMPLn', 'nIMPLp', 'nIMPLn',
              'pANDpIMPLp', 'pANDpIMPLn', 'pANDnIMPLp', 'pANDnIMPLn', 'nANDnIMPLp', 'nANDnIMPLn',
              'pIMPLpORp', 'pIMPLpORn', 'pIMPLnORn', 'nIMPLpORp', 'nIMPLpORn', 'nIMPLnORn',
              'pDIFFp']
              
# dicitionary mapping rule types to desired output string
rule_strings = {'pIMPLp' : '{0} IMPLIES {1}', 'pIMPLn' : '{0} IMPLIES (NOT {1})', 
                'nIMPLp' : '(NOT {0}) IMPLIES {1}', 'nIMPLn' : '(NOT {0}) IMPLIES (NOT {1})',
                'pANDpIMPLp' : '{0} AND {1} IMPLIES {2}', 'pANDpIMPLn' : '{0} AND {1} IMPLIES (NOT {2})', 
                'pANDnIMPLp' : '{0} AND (NOT {1}) IMPLIES {2}', 'pANDnIMPLn' : '{0} AND (NOT {1}) IMPLIES (NOT {2})', 
                'nANDnIMPLp' : '(NOT {0}) AND (NOT {1}) IMPLIES {2}', 'nANDnIMPLn' : '(NOT {0}) AND (NOT {1}) IMPLIES (NOT {2})',
                'pIMPLpORp' : '{0} IMPLIES {1} OR {2}', 'pIMPLpORn' : '{0} IMPLIES {1} OR (NOT {2})', 
                'pIMPLnORn' : '{0} IMPLIES (NOT {1}) OR (NOT {2})', 
                'nIMPLpORp' : '(NOT {0}) IMPLIES {1} OR {2}', 'nIMPLpORn' : '(NOT {0}) IMPLIES {1} OR (NOT {2})', 
                'nIMPLnORn' : '(NOT {0}) IMPLIES (NOT {1}) OR (NOT {2})',
                'pDIFFp' : '{0} DIFFERENT FROM {1}'}              

# dictionary for the confidence values extracted from the data set
# maps from rule type to a list containing rules and their confidence
rules = { }
for rule_type in rule_types:
    rules[rule_type] = {}

def store_rule(rule_name, support, probability, concepts):
    """Stores the given rule in the dictionary (converting probability to confidence)."""
    confidence = (probability - 0.5) * 2
    tuple_name = '_'.join(concepts)
    relative_support = support / len(all_classifications)
    rules[rule_name][tuple_name] = [relative_support, confidence] + concepts

for first_concept in all_concepts:
    if not args.quiet:
        print('\t{0}'.format(first_concept))
    
    first_idx = all_concepts.index(first_concept)
    
    # do the counts
    count_p = np.sum(all_classifications[:,first_idx])
    count_n = len(all_classifications) - count_p

    # create subsets
    classifications_p = all_classifications[all_classifications[:,first_idx] == 1]
    classifications_n = all_classifications[all_classifications[:,first_idx] == 0]

    for second_concept in all_concepts:
        
        if first_concept == second_concept: # ignore trivial rules
            continue
        
        second_idx = all_concepts.index(second_concept)        
        tuple_name = "{0}_{1}".format(first_concept, second_concept)

        # do the counts
        count_pp = np.sum(classifications_p[:,second_idx])
        count_pn = len(classifications_p) - count_pp
        
        count_np = np.sum(classifications_n[:,second_idx])
        count_nn = len(classifications_n) - count_np
 
        # compute the probabilities
        if count_p > 0:
            p_impl_p = count_pp / count_p
            p_impl_n = count_pn / count_p
            
            store_rule('pIMPLp', count_p, p_impl_p, [first_concept, second_concept])
            store_rule('pIMPLn', count_p, p_impl_n, [first_concept, second_concept])
            
        if count_n > 0:
            n_impl_p = count_np / count_n
            n_impl_n = count_nn / count_n

            store_rule('nIMPLp', count_n, n_impl_p, [first_concept, second_concept])
            store_rule('nIMPLn', count_n, n_impl_n, [first_concept, second_concept])

        # create subsets
        classifications_pp = classifications_p[classifications_p[:,second_idx] == 1]
        classifications_pn = classifications_p[classifications_p[:,second_idx] == 0]
        
        classifications_np = classifications_n[classifications_n[:,second_idx] == 1]
        classifications_nn = classifications_n[classifications_n[:,second_idx] == 0]
            
        
        for third_concept in all_concepts:
            
            if third_concept in [first_concept, second_concept]: # ignore trivial rules
                continue
            
            third_idx = all_concepts.index(third_concept)
            triple_name = "{0}_{1}_{2}".format(first_concept, second_concept, third_concept)
            
            # do the counts
            count_ppp = np.sum(classifications_pp[:,third_idx])
            count_ppn = len(classifications_pp) - count_ppp
            
            count_pnp = np.sum(classifications_pn[:,third_idx])
            count_pnn = len(classifications_pn) - count_pnp
            
            count_npp = np.sum(classifications_np[:,third_idx])
            count_npn = len(classifications_np) - count_npp
            
            count_nnp = np.sum(classifications_nn[:,third_idx])
            count_nnn = len(classifications_nn) - count_nnp
            
            # compute the probabilities            
            
            # take care of "A AND B IMPLIES C"  
            if count_pp > 0:
                p_and_p_impl_p = count_ppp / count_pp
                p_and_p_impl_n = count_ppn / count_pp

                store_rule('pANDpIMPLp', count_pp, p_and_p_impl_p, [first_concept, second_concept, third_concept])
                store_rule('pANDpIMPLn', count_pp, p_and_p_impl_n, [first_concept, second_concept, third_concept])
                
            if count_pn > 0:
                p_and_n_impl_p = count_pnp / count_pn
                p_and_n_impl_n = count_pnn / count_pn
            
                store_rule('pANDnIMPLp', count_pn, p_and_n_impl_p, [first_concept, second_concept, third_concept])
                store_rule('pANDnIMPLn', count_pn, p_and_n_impl_n, [first_concept, second_concept, third_concept])

            if count_nn > 0:
                n_and_n_impl_p = count_nnp / count_nn
                n_and_n_impl_n = count_nnn / count_nn
                       
                store_rule('nANDnIMPLp', count_nn, n_and_n_impl_p, [first_concept, second_concept, third_concept])
                store_rule('nANDnIMPLn', count_nn, n_and_n_impl_n, [first_concept, second_concept, third_concept])
            
            # take care of "A IMPLIES B OR C"
            if count_p > 0:
                # size(B or C) = size(B) + size(not B and C) 
                p_impl_p_or_p = (count_pp + count_pnp) / count_p 
                # size(B or not C) = size(B) + size(not B and not C)
                p_impl_p_or_n = (count_pp + count_pnn) / count_p   
                # size(not B or not C) = size(not B) + size(B and not C)
                p_impl_n_or_n = (count_pn + count_ppn) / count_p    
            
                store_rule('pIMPLpORp', count_p, p_impl_p_or_p, [first_concept, second_concept, third_concept])
                store_rule('pIMPLpORn', count_p, p_impl_p_or_n, [first_concept, second_concept, third_concept])
                store_rule('pIMPLnORn', count_p, p_impl_n_or_n, [first_concept, second_concept, third_concept])
            
            if count_n > 0:
                # size(B or C) = size(B) + size(not B and C) 
                n_impl_p_or_p = (count_np + count_nnp) / count_n  
                # size(B or not C) = size(B) + size(not B and not C)
                n_impl_p_or_n = (count_np + count_nnn) / count_n    
                # size(not B or not C) = size(not B) + size(B and not C)
                n_impl_n_or_n = (count_nn + count_npn) / count_n    
            
                store_rule('nIMPLpORp', count_n, n_impl_p_or_p, [first_concept, second_concept, third_concept])
                store_rule('nIMPLpORn', count_n, n_impl_p_or_n, [first_concept, second_concept, third_concept])
                store_rule('nIMPLnORn', count_n, n_impl_n_or_n, [first_concept, second_concept, third_concept])

central_output = {'rule_types' : rule_types, 'rule_strings' : rule_strings}

if not args.quiet:
    print('storing central pickle file')
    
with open(os.path.join(args.output_folder, 'central.pickle'), 'wb') as f:
    pickle.dump(central_output, f)

# store in individual pickle files for memory reasons
for rule_type in rule_types:
    if not args.quiet:
        print('storing {0} pickle file'.format(rule_type))
    with open(os.path.join(args.output_folder, '{0}.pickle'.format(rule_type)), 'wb') as f:
        pickle.dump(rules[rule_type], f)