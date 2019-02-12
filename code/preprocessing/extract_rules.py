# -*- coding: utf-8 -*-
"""
Extracts rules from data set by counting labels.

Created on Mon Feb 11 15:04:57 2019

@author: lbechberger
"""

import argparse, os
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
    print("Counting co-occurrences ...")

# store the counts in dictionaries
count_p = {}    # first_concept is positive
count_n = {}    # first_concept is negative
count_pp = {}   # first concept is positive, second concept is positive
count_pn = {}   # first concept is positive, second concept is negative
count_np = {}   # first concept is negative, second concept is positive
count_nn = {}   # first concept is negative, second concept is negative
count_ppp = {}  # first concept is positive, second concept is positive, third concept is positive
count_ppn = {}  # ...
count_pnp = {}
count_pnn = {}
count_npp = {}
count_npn = {}
count_nnp = {}
count_nnn = {}

# collect all the counts    
for first_concept in all_concepts:
    if not args.quiet:
        print('\t{0}'.format(first_concept))
    
    first_idx = all_concepts.index(first_concept)
    
    count_p[first_concept] = np.sum(all_classifications[:,first_idx])
    count_n[first_concept] = len(all_classifications) - count_p[first_concept]

    classifications_p = all_classifications[all_classifications[:,first_idx] == 1]
    classifications_n = all_classifications[all_classifications[:,first_idx] == 0]

    for second_concept in all_concepts:
        
        if first_concept == second_concept: # ignore trivial rules
            continue
        
        second_idx = all_concepts.index(second_concept)        
        tuple_name = "{0}_{1}".format(first_concept, second_concept)

        count_pp[tuple_name] = np.sum(classifications_p[:,second_idx])
        count_pn[tuple_name] = len(classifications_p) - count_pp[tuple_name]
        
        count_np[tuple_name] = np.sum(classifications_n[:,second_idx])
        count_nn[tuple_name] = len(classifications_n) - count_np[tuple_name]
        
        classifications_pp = classifications_p[classifications_p[:,second_idx] == 1]
        classifications_pn = classifications_p[classifications_p[:,second_idx] == 0]
        
        classifications_np = classifications_n[classifications_n[:,second_idx] == 1]
        classifications_nn = classifications_n[classifications_n[:,second_idx] == 0]
            
        
        for third_concept in all_concepts:
            
            if third_concept in [first_concept, second_concept]: # ignore trivial rules
                continue
            
            third_idx = all_concepts.index(third_concept)
            triple_name = "{0}_{1}_{2}".format(first_concept, second_concept, third_concept)
            
            count_ppp[triple_name] = np.sum(classifications_pp[:,third_idx])
            count_ppn[triple_name] = len(classifications_pp) - count_ppp[triple_name]
            
            count_pnp[triple_name] = np.sum(classifications_pn[:,third_idx])
            count_pnn[triple_name] = len(classifications_pn) - count_pnp[triple_name]
            
            count_npp[triple_name] = np.sum(classifications_np[:,third_idx])
            count_npn[triple_name] = len(classifications_np) - count_npp[triple_name]
            
            count_nnp[triple_name] = np.sum(classifications_nn[:,third_idx])
            count_nnn[triple_name] = len(classifications_nn) - count_nnp[triple_name]
            

if not args.quiet:
    print("Extracting rule probabilities...")
            
# now look for rules like "first_concept IMPLIES second_concept"
# and "first_concept AND second_concept IMPLIES third_concept" and "first_concept IMPLIES second_concept OR third_concept"
for first_concept in all_concepts:
    if not args.quiet:
        print('\t{0}'.format(first_concept))
    
    for second_concept in all_concepts:
        
        if first_concept == second_concept: # ignore trivial rules
            continue

        tuple_name = "{0}_{1}".format(first_concept, second_concept)        
        
        if count_p[first_concept] > 0:
            p_impl_p = count_pp[tuple_name] / count_p[first_concept]
            p_impl_n = count_pn[tuple_name] / count_p[first_concept]
            
            rules['pIMPLp'].append([p_impl_p, first_concept, second_concept])        
            rules['pIMPLn'].append([p_impl_n, first_concept, second_concept])        

        if count_n[first_concept] > 0:
            n_impl_p = count_np[tuple_name] / count_n[first_concept]
            n_impl_n = count_nn[tuple_name] / count_n[first_concept]

            rules['nIMPLn'].append([n_impl_n, first_concept, second_concept])     
            rules['nIMPLp'].append([n_impl_p, first_concept, second_concept])        

        for third_concept in all_concepts:
            
            if third_concept in [first_concept, second_concept]: # ignore trivial rules
                continue
            
            triple_name = "{0}_{1}_{2}".format(first_concept, second_concept, third_concept)
            
            # take care of "A AND B IMPLIES C"  
            if count_pp[tuple_name] > 0:
                p_and_p_impl_p = count_ppp[triple_name] / count_pp[tuple_name]
                p_and_p_impl_n = count_ppn[triple_name] / count_pp[tuple_name]

                rules['pANDpIMPLp'].append([p_and_p_impl_p, first_concept, second_concept, third_concept])
                rules['pANDpIMPLn'].append([p_and_p_impl_n, first_concept, second_concept, third_concept])

            if count_pn[tuple_name] > 0:
                p_and_n_impl_p = count_pnp[triple_name] / count_pn[tuple_name]
                p_and_n_impl_n = count_pnn[triple_name] / count_pn[tuple_name]
            
                rules['pANDnIMPLp'].append([p_and_n_impl_p, first_concept, second_concept, third_concept])
                rules['pANDnIMPLn'].append([p_and_n_impl_n, first_concept, second_concept, third_concept])

            if count_np[tuple_name] > 0:
                n_and_p_impl_p = count_npp[triple_name] / count_np[tuple_name]
                n_and_p_impl_n = count_npn[triple_name] / count_np[tuple_name]

                rules['nANDpIMPLp'].append([n_and_p_impl_p, first_concept, second_concept, third_concept])
                rules['nANDpIMPLn'].append([n_and_p_impl_n, first_concept, second_concept, third_concept])

            if count_nn[tuple_name] > 0:
                n_and_n_impl_p = count_nnp[triple_name] / count_nn[tuple_name]
                n_and_n_impl_n = count_nnn[triple_name] / count_nn[tuple_name]
                       
                rules['nANDnIMPLp'].append([n_and_n_impl_p, first_concept, second_concept, third_concept])
                rules['nANDnIMPLn'].append([n_and_n_impl_n, first_concept, second_concept, third_concept])
            
            # take care of "A IMPLIES B OR C"
            if count_p[first_concept] > 0:
                # size(B or C) = size(B) + size(not B and C) 
                p_impl_p_or_p = (count_pp[tuple_name] + count_pnp[triple_name]) / count_p[first_concept] 
                # size(B or not C) = size(B) + size(not B and not C)
                p_impl_p_or_n = (count_pp[tuple_name] + count_pnn[triple_name]) / count_p[first_concept]   
                # size(not B or C) = size(not B) + size(B and C)
                p_impl_n_or_p = (count_pn[tuple_name] + count_ppp[triple_name]) / count_p[first_concept]  
                # size(not B or not C) = size(not B) + size(B and not C)
                p_impl_n_or_n = (count_pn[tuple_name] + count_ppn[triple_name]) / count_p[first_concept]    
            
                rules['pIMPLpORp'].append([p_impl_p_or_p, first_concept, second_concept, third_concept])
                rules['pIMPLpORn'].append([p_impl_p_or_n, first_concept, second_concept, third_concept])
                rules['pIMPLnORp'].append([p_impl_n_or_p, first_concept, second_concept, third_concept])
                rules['pIMPLnORn'].append([p_impl_n_or_n, first_concept, second_concept, third_concept])
            
            if count_n[first_concept] > 0:
                # size(B or C) = size(B) + size(not B and C) 
                n_impl_p_or_p = (count_np[tuple_name] + count_nnp[triple_name]) / count_n[first_concept]  
                # size(B or not C) = size(B) + size(not B and not C)
                n_impl_p_or_n = (count_np[tuple_name] + count_nnn[triple_name]) / count_n[first_concept]    
                # size(not B or C) = size(not B) + size(B and C)
                n_impl_n_or_p = (count_nn[tuple_name] + count_npp[triple_name]) / count_n[first_concept]    
                # size(not B or not C) = size(not B) + size(B and not C)
                n_impl_n_or_n = (count_nn[tuple_name] + count_npn[triple_name]) / count_n[first_concept]    
            
                rules['nIMPLpORp'].append([n_impl_p_or_p, first_concept, second_concept, third_concept])
                rules['nIMPLpORn'].append([n_impl_p_or_n, first_concept, second_concept, third_concept])
                rules['nIMPLnORp'].append([n_impl_n_or_p, first_concept, second_concept, third_concept])
                rules['nIMPLnORn'].append([n_impl_n_or_n, first_concept, second_concept, third_concept])

if not args.quiet:
    print("Evaluating the results...")

evaluate_rules(rules, 'genres_and_keywords', 'all', 'counting', args.quiet)

if not args.quiet:
    print("DONE")
