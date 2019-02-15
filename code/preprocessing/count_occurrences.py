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
parser.add_argument('input_file', help = 'path to the file containing the data set information')
parser.add_argument('-o', '--output_folder', help = 'path to output folder for the pickle files', default = '.')                
parser.add_argument('-q', '--quiet', action="store_true", help = 'disables info output')                    
args = parser.parse_args()

# load data set
data_set = pickle.load(open(args.input_file, 'rb'))

all_concepts = data_set['all_concepts']
all_classifications = data_set['all_classifications']

# rule types we're interested in. "p" means "positive", "n" means "negative", 
#"IMPL" means "implies", and "DIFF" means "is different from"
rule_types = ['pIMPLp', 'nIMPLp', 
              'pANDpIMPLp', 'pANDnIMPLp', 'nANDnIMPLp',
              'pIMPLpORp', 'nIMPLpORp']
              
# dicitionary mapping rule types to desired output string
rule_strings = {'pIMPLp' : '{0} IMPLIES {1}',
                'nIMPLp' : '(NOT {0}) IMPLIES {1}',
                'pANDpIMPLp' : '{0} AND {1} IMPLIES {2}',
                'pANDnIMPLp' : '{0} AND (NOT {1}) IMPLIES {2}',
                'nANDnIMPLp' : '(NOT {0}) AND (NOT {1}) IMPLIES {2}',
                'pIMPLpORp' : '{0} IMPLIES {1} OR {2}',
                'nIMPLpORp' : '(NOT {0}) IMPLIES {1} OR {2}'}              

# dictionary for the confidence values extracted from the data set
# maps from rule type to a list containing rules and their confidence
rules = { }
for rule_type in rule_types:
    rules[rule_type] = {}

def store_rule(rule_name, support, probability, concepts):
    """Stores the given rule in the dictionary (converting probability to confidence)."""
    tuple_name = '_'.join(concepts)
    relative_support = support / len(all_classifications)
    rules[rule_name][tuple_name] = [relative_support, probability] + concepts

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
            store_rule('pIMPLp', count_p, p_impl_p, [first_concept, second_concept])
            
        if count_n > 0:
            n_impl_p = count_np / count_n
            store_rule('nIMPLp', count_n, n_impl_p, [first_concept, second_concept])

        # create subsets
        classifications_pp = classifications_p[classifications_p[:,second_idx] == 1]
        classifications_pn = classifications_p[classifications_p[:,second_idx] == 0]
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
            
            count_nnp = np.sum(classifications_nn[:,third_idx])
            count_nnn = len(classifications_nn) - count_nnp
            
            # compute the probabilities - only for well-ordered pairs (in order to avoid duplications)           
            
            if first_concept < second_concept:
                # take care of "A AND B IMPLIES C"  
                if count_pp > 0:
                    p_and_p_impl_p = count_ppp / count_pp
                    store_rule('pANDpIMPLp', count_pp, p_and_p_impl_p, [first_concept, second_concept, third_concept])
                    
                if count_pn > 0:
                    p_and_n_impl_p = count_pnp / count_pn
                    store_rule('pANDnIMPLp', count_pn, p_and_n_impl_p, [first_concept, second_concept, third_concept])
    
                if count_nn > 0:
                    n_and_n_impl_p = count_nnp / count_nn
                    store_rule('nANDnIMPLp', count_nn, n_and_n_impl_p, [first_concept, second_concept, third_concept])
            
            if second_concept < third_concept:
                # take care of "A IMPLIES B OR C"
                if count_p > 0:
                    # size(B or C) = size(B) + size(not B and C) 
                    p_impl_p_or_p = (count_pp + count_pnp) / count_p 
                    store_rule('pIMPLpORp', count_p, p_impl_p_or_p, [first_concept, second_concept, third_concept])
                
                if count_n > 0:
                    # size(B or C) = size(B) + size(not B and C) 
                    n_impl_p_or_p = (count_np + count_nnp) / count_n  
                    store_rule('nIMPLpORp', count_n, n_impl_p_or_p, [first_concept, second_concept, third_concept])

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