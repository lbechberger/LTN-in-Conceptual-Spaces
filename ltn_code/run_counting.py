# -*- coding: utf-8 -*-
"""
Runs the label counting algorithm in order to extract rules.

Created on Thu Oct 18 11:03:27 2018

@author: lbechberger
"""

import random, argparse
import util
import numpy as np

# fix random seed to ensure reproducibility
random.seed(42)

# parse command line arguments
parser = argparse.ArgumentParser(description='rule extraction by label counting')
parser.add_argument('-q', '--quiet', action="store_true",
                    help = 'disables info output')                    
parser.add_argument('config_file', help = 'the config file to use')
parser.add_argument('config_name', help = 'the name of the configuration')
args = parser.parse_args()

# read config file
config = util.parse_config_file(args.config_file, args.config_name)
             
# dictionary for the confidence values extracted from the data set
# maps from rule type to a list containing rules and their confidence
rules = { }
for rule_type in util.rule_types:
    rules[rule_type] = []

if not args.quiet:
    print("Looking for rules involving two concepts...")
    
# first look for simple rules involving only two concepts
# like "first_concept IMPLIES second_concept" and "first_concept IS DIFFERENT FROM second_concept"
for first_concept in config["concepts"]:
    if not args.quiet:
        print(first_concept)
    for second_concept in config["concepts"]:
        
        if first_concept == second_concept:
            continue
        
        # for all counters: first entry is training set, second entry is validation set, third entry is test set
        # 'p' stands for 'positive', 'n' for 'negative'
        count_p = np.array([0, 0, 0])       # first_concept is positive
        count_pp = np.array([0, 0, 0])      # first concept is positive, second concept is positive
        count_n = np.array([0, 0, 0])       # first_concept is negative
        count_np = np.array([0, 0, 0])      # first concept is negative, second concept is positive
        
        for count_idx, data_set in enumerate(['training_vectors', 'validation_vectors', 'test_vectors']):
            for (labels, _) in config[data_set]:
                if first_concept in labels:
                    count_p[count_idx] += 1
                    if second_concept in labels:
                        count_pp[count_idx] += 1
                else:
                    count_n[count_idx] += 1
                    if second_concept in labels:
                        count_np[count_idx] += 1
        
        # division and subtraction are taking place element-wise thanks to numpy  
        if all(x > 0 for x in count_p):
            p_impl_p = count_pp / count_p
            p_impl_n = 1 - p_impl_p
            
            rules['pIMPLp'].append([p_impl_p, first_concept, second_concept])        
            rules['pIMPLn'].append([p_impl_n, first_concept, second_concept])        

        if all(x > 0 for x in count_n):
            n_impl_p = count_np / count_n
            n_impl_n = 1 - n_impl_p

            rules['nIMPLn'].append([n_impl_n, first_concept, second_concept])     
            rules['nIMPLp'].append([n_impl_p, first_concept, second_concept])        

        if all(x > 0 for x in (count_p + count_np)):
            jaccard_distance = 1 - (count_pp / (count_p + count_np))
            rules['pDIFFp'].append([jaccard_distance, first_concept, second_concept])

if not args.quiet:
    print("Looking for rules involving three concepts...")

# now look for rules involving three concepts
# like "first_concept AND second_concept IMPLIES third_concept" or "first_concept IMPLIES second_concept OR third_concept"
for first_concept in config["concepts"]:
    if not args.quiet:
        print(first_concept)
    for second_concept in config["concepts"]:
        for third_concept in config["concepts"]:
           
            if first_concept == second_concept or first_concept == third_concept or second_concept == third_concept:
                continue
            
            # for all counters: first entry is training set, second entry is validation set, third entry is test set
            # 'p' stands for 'positive', 'n' for 'negative'
            count_p   = np.array([0, 0, 0])     # first concept is positive
            count_pp  = np.array([0, 0, 0])     # first concept is positive, second concept is positive
            count_ppp = np.array([0, 0, 0])     # first concept is positive, second concept is positive, third concept is positive
            count_ppn = np.array([0, 0, 0])     # first concept is positive, second concept is positive, third concept is negative
            count_pn  = np.array([0, 0, 0])     # first concept is positive, second concept is negative
            count_pnp = np.array([0, 0, 0])     # first concept is positive, second concept is negative, third concept is positive
            count_pnn = np.array([0, 0, 0])     # first concept is positive, second concept is negative, third concept is negative
            count_n   = np.array([0, 0, 0])     # first_concept is negative
            count_np  = np.array([0, 0, 0])     # first concept is negative, second concept is positive
            count_npp = np.array([0, 0, 0])     # first concept is negative, second concept is positive, third concept is positive
            count_npn = np.array([0, 0, 0])     # first concept is negative, second concept is positive, third concept is negative
            count_nn  = np.array([0, 0, 0])     # first concept is negative, second concept is negative
            count_nnp = np.array([0, 0, 0])     # first concept is negative, second concept is negative, third concept is positive
            count_nnn = np.array([0, 0, 0])     # first concept is negative, second concept is negative, third concept is negative
            
            for count_idx, data_set in enumerate(['training_vectors', 'validation_vectors', 'test_vectors']):
                for (labels, _) in config[data_set]:
                    if first_concept in labels:
                        count_p[count_idx] += 1
                        if second_concept in labels:
                            count_pp[count_idx] += 1
                            if third_concept in labels:
                                count_ppp[count_idx] += 1
                            else:
                                count_ppn[count_idx] += 1
                        else:
                            count_pn[count_idx] += 1
                            if third_concept in labels:
                                count_pnp[count_idx] += 1
                            else:
                                count_pnn[count_idx] += 1
                    else:
                        count_n[count_idx] += 1
                        if second_concept in labels:
                            count_np[count_idx] += 1
                            if third_concept in labels:
                                count_npp[count_idx] += 1
                            else:
                                count_npn[count_idx] += 1
                        else:
                            count_nn[count_idx] += 1
                            if third_concept in labels:
                                count_nnp[count_idx] += 1
                            else:
                                count_nnn[count_idx] += 1
            
            # take care of "A AND B IMPLIES C"  
            if all(x > 0 for x in count_pp):
                p_and_p_impl_p = count_ppp / count_pp
                p_and_p_impl_n = count_ppn / count_pp

                rules['pANDpIMPLp'].append([p_and_p_impl_p, first_concept, second_concept, third_concept])
                rules['pANDpIMPLn'].append([p_and_p_impl_n, first_concept, second_concept, third_concept])

            if all(x > 0 for x in count_pn):
                p_and_n_impl_p = count_pnp / count_pn
                p_and_n_impl_n = count_pnn / count_pn
            
                rules['pANDnIMPLp'].append([p_and_n_impl_p, first_concept, second_concept, third_concept])
                rules['pANDnIMPLn'].append([p_and_n_impl_n, first_concept, second_concept, third_concept])

            if all(x > 0 for x in count_np):
                n_and_p_impl_p = count_npp / count_np
                n_and_p_impl_n = count_npn / count_np

                rules['nANDpIMPLp'].append([n_and_p_impl_p, first_concept, second_concept, third_concept])
                rules['nANDpIMPLn'].append([n_and_p_impl_n, first_concept, second_concept, third_concept])

            if all(x > 0 for x in count_nn):
                n_and_n_impl_p = count_nnp / count_nn
                n_and_n_impl_n = count_nnn / count_nn
                       
                rules['nANDnIMPLp'].append([n_and_n_impl_p, first_concept, second_concept, third_concept])
                rules['nANDnIMPLn'].append([n_and_n_impl_n, first_concept, second_concept, third_concept])
            
            # take care of "A IMPLIES B OR C"
            if all(x > 0 for x in count_p):
                p_impl_p_or_p = (count_pp + count_pnp) / count_p    # size(B or C) = size(B) + size(not B and C) 
                p_impl_p_or_n = (count_pp + count_pnn) / count_p    # size(B or not C) = size(B) + size(not B and not C)
                p_impl_n_or_p = (count_pn + count_ppp) / count_p    # size(not B or C) = size(not B) + size(B and C)
                p_impl_n_or_n = (count_pn + count_ppn) / count_p    # size(not B or not C) = size(not B) + size(B and not C)
            
                rules['pIMPLpORp'].append([p_impl_p_or_p, first_concept, second_concept, third_concept])
                rules['pIMPLpORn'].append([p_impl_p_or_n, first_concept, second_concept, third_concept])
                rules['pIMPLnORp'].append([p_impl_n_or_p, first_concept, second_concept, third_concept])
                rules['pIMPLnORn'].append([p_impl_n_or_n, first_concept, second_concept, third_concept])
            
            if all(x > 0 for x in count_n):
                n_impl_p_or_p = (count_np + count_nnp) / count_n    # size(B or C) = size(B) + size(not B and C) 
                n_impl_p_or_n = (count_np + count_nnn) / count_n    # size(B or not C) = size(B) + size(not B and not C)
                n_impl_n_or_p = (count_nn + count_npp) / count_n    # size(not B or C) = size(not B) + size(B and C)
                n_impl_n_or_n = (count_nn + count_npn) / count_n    # size(not B or not C) = size(not B) + size(B and not C)
            
                rules['nIMPLpORp'].append([n_impl_p_or_p, first_concept, second_concept, third_concept])
                rules['nIMPLpORn'].append([n_impl_p_or_n, first_concept, second_concept, third_concept])
                rules['nIMPLnORp'].append([n_impl_n_or_p, first_concept, second_concept, third_concept])
                rules['nIMPLnORn'].append([n_impl_n_or_n, first_concept, second_concept, third_concept])

if not args.quiet:
    print("Evaluating the results...")

util.evaluate_rules(rules, args.config_file.split('.')[0], args.config_name, 'counting', args.quiet)

if not args.quiet:
    print("DONE")
