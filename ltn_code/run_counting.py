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

# define names of output files
summary_output_file = "output/{0}_{1}-rules.csv".format(args.config_file.split('.')[0], args.config_name)
rules_output_prefix = "output/rules/{0}-{1}".format(args.config_file.split('.')[0], args.config_name)
rules_output_template = "{0}-{1}-{2}.csv"

# read config file
config = util.parse_config_file(args.config_file, args.config_name)

# rule types we're interested in. "p" means "positive", "n" means "negative", 
#"IMPL" means "implies", and "DIFF" means "is different from"
rule_types = ['pIMPLp', 'pIMPLn', 'nIMPLp', 'nIMPLn',
              'pANDpIMPLp', 'pANDpIMPLn', 'pANDnIMPLp', 'pANDnIMPLn', 'nANDpIMPLp', 'nANDpIMPLn', 'nANDnIMPLp', 'nANDnIMPLn',
              'pIMPLpORp', 'pIMPLpORn', 'pIMPLnORp', 'pIMPLnORn', 'nIMPLpORp', 'nIMPLpORn', 'nIMPLnORp', 'nIMPLnORn',
              'pDIFFp']

# dicitionary mapping rule types to desired output string
rule_strings = {'pIMPLp' : '{0} IMPLIES {1}', 'pIMPLn' : '{0} IMPLIES (NOT {1})', 
                'nIMPLp' : '(NOT {0}) IMPLIES {1}', 'nIMPLn' : '(NOT {0}) IMPLIES (NOT {1})',
                'pANDpIMPLp' : '{0} AND {1} IMPLIES {2}', 'pANDpIMPLn' : '{0} AND {1} IMPLIES (NOT {2})', 
                'pANDnIMPLp' : '{0} AND (NOT {1}) IMPLIES {2}', 'pANDnIMPLn' : '{0} AND (NOT {1}) IMPLIES (NOT {2})', 
                'nANDpIMPLp' : '(NOT {0}) AND {1} IMPLIES {2}', 'nANDpIMPLn' : '(NOT {0}) AND {1} IMPLIES (NOT {2})', 
                'nANDnIMPLp' : '(NOT {0}) AND (NOT {1}) IMPLIES {2}', 'nANDnIMPLn' : '(NOT {0}) AND (NOT {1}) IMPLIES (NOT {2})',
                'pIMPLpORp' : '{0} IMPLIES {1} OR {2}', 'pIMPLpORn' : '{0} IMPLIES {1} OR (NOT {2})', 
                'pIMPLnORp' : '{0} IMPLIES (NOT {1}) OR {2}', 'pIMPLnORn' : '{0} IMPLIES (NOT {1}) OR (NOT {2})', 
                'nIMPLpORp' : '(NOT {0}) IMPLIES {1} OR {2}', 'nIMPLpORn' : '(NOT {0}) IMPLIES {1} OR (NOT {2})', 
                'nIMPLnORp' : '(NOT {0}) IMPLIES (NOT {1}) OR {2}', 'nIMPLnORn' : '(NOT {0}) IMPLIES (NOT {1}) OR (NOT {2})',
                'pDIFFp' : '{0} DIFFERENT FROM {1}'}              
              
# dictionary for the confidence values extracted from the training set
# maps from rule type to a list containing rules and their confidence
rules = { }
for rule_type in rule_types:
    rules[rule_type] = []

# first look for simple rules involving only two concepts
# like "first_concept IMPLIES second_concept" and "first_concept IS DIFFERENT FROM second_concept"
for first_concept in config["concepts"]:
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

# now look for rules involving three concepts
# like "first_concept AND second_concept IMPLIES third_concept" or "first_concept IMPLIES second_concept OR third_concept"
for first_concept in config["concepts"]:
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


output_strings = ['rule_type,desired_threshold,training_threshold,num_rules,min_test_accuracy,avg_test_accuracy\n']

for rule_type, rule_instances in rules.items():
    for confidence_threshold in [0.7, 0.8, 0.9, 0.95, 0.99]:
        # filter list of rules according to confidence_threshold
        filtered_list = list(filter(lambda x: x[0][1] >= confidence_threshold and x[0][0] >= confidence_threshold, rule_instances))
        if len(filtered_list) == 0:
            print("Could not find rules of type {0} when trying to achieve {1} on validation set.".format(rule_type, confidence_threshold))
            continue
        
        # compute threshold on training data as well as performance on test data
        training_threshold = min(map(lambda x: x[0][0], filtered_list))
        average_test_accuracy = sum(map(lambda x: x[0][2], filtered_list)) / len(filtered_list)
        minimal_test_accuracy = min(map(lambda x: x[0][2], filtered_list))

        # print right away        
        print("Threshold for rules of type {0} when trying to achieve {1} on validation set: {2} " \
                "(leaving {3} rules, accuracy on test set: avg {4}, min {5})".format(rule_type, confidence_threshold, 
                  training_threshold, len(filtered_list), average_test_accuracy, minimal_test_accuracy))
        
        # store for output into file
        output_strings.append(",".join([rule_type, str(confidence_threshold), str(training_threshold), str(len(filtered_list)), 
                                        str(minimal_test_accuracy), str(average_test_accuracy)]) + '\n')
        
        # write resulting rules into file
        rules_file_name =  rules_output_template.format(rules_output_prefix, rule_type, confidence_threshold)
        with open(rules_file_name, 'w') as f:
            f.write("rule,training,validation,test\n")
            for rule in filtered_list:
                if len(rule) == 3:
                    rule_string = rule_strings[rule_type].format(rule[1], rule[2])
                elif len(rule) == 4:
                    rule_string = rule_strings[rule_type].format(rule[1], rule[2], rule[3])
                else:
                    raise(Exception("invalid length of rule information"))
                line = "{0},{1},{2},{3}\n".format(rule_string, rule[0][0], rule[0][1], rule[0][2])
                f.write(line)
        
    print("")

with open(summary_output_file, 'w') as f:
    for line in output_strings:
        f.write(line)