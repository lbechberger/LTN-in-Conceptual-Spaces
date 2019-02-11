# -*- coding: utf-8 -*-
"""
Extracts rules from data set by counting labels.

Created on Mon Feb 11 15:04:57 2019

@author: lbechberger
"""

import argparse, os, itertools
import numpy as np

# parse command line arguments
parser = argparse.ArgumentParser(description='rule extraction by label counting')                  
parser.add_argument('keywords_folder', help = 'path to the folder containing label information about plot keywords')
parser.add_argument('genres_folder', help = 'path to the folder containing label information about genres')
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

# define names of output files
summary_output_file_template = "output/{0}_{1}-rules_{2}.csv"
rules_output_prefix_template = "output/rules_{2}/{0}-{1}"
rules_output_template = "{0}-{1}-{2}.csv"

def evaluate_rules(rules_dict, data_set, config, algorithm, quiet):
    """Evaluates the validity of the rules given in the rules_dict and outputs them (both to console and files)."""

    summary_output_file = summary_output_file_template.format(data_set, config, algorithm)
    rules_output_prefix = rules_output_prefix_template.format(data_set, config, algorithm)
    
    output_strings = ['rule_type,desired_threshold,num_rules\n']

    if not quiet:
        print("\nRule evaluation")
    
    for rule_type, rule_instances in rules_dict.items():
        for confidence_threshold in [0.7, 0.8, 0.9, 0.95, 0.99]:
            # filter list of rules according to confidence_threshold
            filtered_list = list(filter(lambda x: x[0] >= confidence_threshold , rule_instances))
            if len(filtered_list) == 0:
                if not quiet:
                    print("Could not find rules of type {0} when trying to achieve {1} on validation set.".format(rule_type, confidence_threshold))
                continue
            
            # print right away        
            if not quiet:
                print("Number of rules of type {0} left when trying to achieve {1} on data set: {2} ".format(rule_type, confidence_threshold, 
                      len(filtered_list)))
            
            # store for output into file
            output_strings.append(",".join([rule_type, str(confidence_threshold), str(len(filtered_list))]) + '\n')
            
            # write resulting rules into file
            rules_file_name =  rules_output_template.format(rules_output_prefix, rule_type, confidence_threshold)
            with open(rules_file_name, 'w') as f:
                f.write("rule,confidence\n")
                for rule in filtered_list:
                    if len(rule) == 3:
                        rule_string = rule_strings[rule_type].format(rule[1], rule[2])
                    elif len(rule) == 4:
                        rule_string = rule_strings[rule_type].format(rule[1], rule[2], rule[3])
                    else:
                        raise(Exception("invalid length of rule information"))
                    line = "{0},{1}\n".format(rule_string, rule[0])
                    f.write(line)
        
        if not quiet:
            print("")
    
    with open(summary_output_file, 'w') as f:
        for line in output_strings:
            f.write(line)    


# dictionary for the confidence values extracted from the data set
# maps from rule type to a list containing rules and their confidence
rules = { }
for rule_type in rule_types:
    rules[rule_type] = []

if not args.quiet:
    print("Looking for rules involving two concepts...")
    
# first look for simple rules involving only two concepts
# like "first_concept IMPLIES second_concept" and "first_concept IS DIFFERENT FROM second_concept"
for first_concept in all_concepts:
    if not args.quiet:
        print(first_concept)
    for second_concept in all_concepts:
        
        if first_concept == second_concept:
            continue
        
        first_idx = all_concepts.index(first_concept)
        second_idx = all_concepts.index(second_concept)        
        
        # for all counters: first entry is training set, second entry is validation set, third entry is test set
        # 'p' stands for 'positive', 'n' for 'negative'
        count_p = 0       # first_concept is positive
        count_pp = 0      # first concept is positive, second concept is positive
        count_n = 0       # first_concept is negative
        count_np = 0      # first concept is negative, second concept is positive
    
        
        for line in all_classifications:
            if line[first_idx] == 1:
                count_p += 1
                if line[second_idx] == 1:
                    count_pp += 1
            else:
                count_n += 1
                if line[second_idx] == 1:
                    count_np += 1
        
        if count_p > 0:
            p_impl_p = count_pp / count_p
            p_impl_n = 1 - p_impl_p
            
            rules['pIMPLp'].append([p_impl_p, first_concept, second_concept])        
            rules['pIMPLn'].append([p_impl_n, first_concept, second_concept])        

        if count_n > 0:
            n_impl_p = count_np / count_n
            n_impl_n = 1 - n_impl_p

            rules['nIMPLn'].append([n_impl_n, first_concept, second_concept])     
            rules['nIMPLp'].append([n_impl_p, first_concept, second_concept])        

        if count_p + count_np > 0:
            jaccard_distance = 1 - (count_pp / (count_p + count_np))
            rules['pDIFFp'].append([jaccard_distance, first_concept, second_concept])

if not args.quiet:
    print("Looking for rules involving three concepts...")

# now look for rules involving three concepts
# like "first_concept AND second_concept IMPLIES third_concept" or "first_concept IMPLIES second_concept OR third_concept"
for first_concept in all_concepts:
    if not args.quiet:
        print(first_concept)
    for second_concept in all_concepts:
        for third_concept in all_concepts:
           
            if first_concept == second_concept or first_concept == third_concept or second_concept == third_concept:
                continue

            first_idx = all_concepts.index(first_concept)
            second_idx = all_concepts.index(second_concept)
            third_idx = all_concepts.index(third_concept)
            
            # for all counters: first entry is training set, second entry is validation set, third entry is test set
            # 'p' stands for 'positive', 'n' for 'negative'
            count_p   = 0     # first concept is positive
            count_pp  = 0     # first concept is positive, second concept is positive
            count_ppp = 0     # first concept is positive, second concept is positive, third concept is positive
            count_ppn = 0     # first concept is positive, second concept is positive, third concept is negative
            count_pn  = 0     # first concept is positive, second concept is negative
            count_pnp = 0     # first concept is positive, second concept is negative, third concept is positive
            count_pnn = 0     # first concept is positive, second concept is negative, third concept is negative
            count_n   = 0     # first_concept is negative
            count_np  = 0     # first concept is negative, second concept is positive
            count_npp = 0     # first concept is negative, second concept is positive, third concept is positive
            count_npn = 0     # first concept is negative, second concept is positive, third concept is negative
            count_nn  = 0     # first concept is negative, second concept is negative
            count_nnp = 0     # first concept is negative, second concept is negative, third concept is positive
            count_nnn = 0     # first concept is negative, second concept is negative, third concept is negative
            
            for line in all_classifications:
                if line[first_idx] == 1:
                    count_p += 1
                    if line[second_idx] == 1:
                        count_pp += 1
                        if line[third_idx] == 1:
                            count_ppp += 1
                        else:
                            count_ppn += 1
                    else:
                        count_pn += 1
                        if line[third_idx] == 1:
                            count_pnp += 1
                        else:
                            count_pnn += 1
                else:
                    count_n += 1
                    if line[second_idx] == 1:
                        count_np += 1
                        if line[third_idx] == 1:
                            count_npp += 1
                        else:
                            count_npn += 1
                    else:
                        count_nn += 1
                        if line[third_idx] == 1:
                            count_nnp += 1
                        else:
                            count_nnn += 1
        
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

evaluate_rules(rules, 'genres_and_keywords', 'all', 'counting', args.quiet)

if not args.quiet:
    print("DONE")
