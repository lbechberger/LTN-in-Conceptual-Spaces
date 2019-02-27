# -*- coding: utf-8 -*-
"""
Extracts rules from data set based on the frequent item sets.

Inspired by https://github.com/rasbt/mlxtend/blob/master/mlxtend/frequent_patterns/association_rules.py

Created on Mon Feb 11 15:04:57 2019

@author: lbechberger
"""

import argparse, pickle, os
from itertools import combinations

# parse command line arguments
parser = argparse.ArgumentParser(description='rule extraction based on label counting')                  
parser.add_argument('input_file', help = 'path to the input pickle file')
parser.add_argument('-o', '--output_folder', help = 'path to output folder for the rule files', default = '.')                
parser.add_argument('-s', '--support', type = float, help = 'threshold for antecedent support', default = 0.01)
parser.add_argument('-c', '--confidence', type = float, help = 'threshold for rule confidence', default = 0.9)                
parser.add_argument('-q', '--quiet', action="store_true", help = 'disables info output')                    
args = parser.parse_args()

# import data
data = pickle.load(open(args.input_file, 'rb'))
frequent_itemsets = data['itemsets']
supports = data['supports']
concepts = data['concepts']
border = data['border']

# prepare data structures
rules = {}
itemset_dict = {}
for size in frequent_itemsets.keys():
    rules[size] = {}
    # dictionary from item set to support
    itemset_dict[size] = dict(zip(map(lambda x: frozenset(x), frequent_itemsets[size]), supports[size]))

# helper function for creating rule string based on antecedent and consequent
def rule_string(antecedent, consequent):
    antecedent_string = ' & '.join(map(lambda x: "{0}(?x)".format(concepts[x]), antecedent))
    consequent_string = ' | '.join(map(lambda x: "{0}(?x)".format(concepts[x]), consequent))
    return "forall ?x: {0} -> {1}".format(antecedent_string, consequent_string)

for size in itemset_dict.keys():
    if size == 1:
        continue # no rules of size 1 needed

    for antecedent_size in range(1, size):
        rules[size][antecedent_size] = {}
    
    # look at each large itemset (i.e., list of literals with the current size)
    for large_itemset, large_support in itemset_dict[size].items():
        # vary the size of the antecedent
        for antecedent_size in range(1, size):
            # pick all the possible antecedents
            for antecedent in combinations(large_itemset, antecedent_size):
            
                consequent = [item for item in large_itemset if item not in antecedent]
                
                if max(consequent) >= border:
                    continue # consequent contains at least one negated literal --> skip
                    
                # calculate support of antecedent    
                antecedent_support = itemset_dict[antecedent_size][frozenset(antecedent)]
                
                # compute numerator for confidence
                # --> use the inclusion-exclusion formula over all the entries of the consequent
                confidence_numerator = 0 
                for l in range(1, len(consequent) + 1):
                    inner_sum = 0.0
                    subsets = list(combinations(consequent, l))           
                    for subset in subsets:
                        modified_subset = frozenset(antecedent + subset)
                        inner_sum += itemset_dict[len(modified_subset)][modified_subset]
                        
                    confidence_numerator += inner_sum * (-1.0)**(l+1)
                
                confidence = confidence_numerator / antecedent_support

                if confidence >= args.confidence and antecedent_support >= args.support:
                    # fulfilled confidence and antecedent support requirements
                
                    rule_novel = True
                    # look for simpler rules by simplifying antecedent
                    if antecedent_size > 1:
                        for smaller_antecedent in combinations(antecedent, antecedent_size - 1):
                            if rule_string(smaller_antecedent, consequent) in rules[size - 1][antecedent_size - 1]:
                                rule_novel = False
                                break
                            
                    # look for simpler rules by simplifying consequent        
                    if rule_novel and len(consequent) > 1:
                        for smaller_consequent in combinations(consequent, len(consequent) - 1):
                            if rule_string(antecedent, smaller_consequent) in rules[size - 1][antecedent_size]:
                                rule_novel = False
                                break
                    
                    if rule_novel:
                        # no simpler rule exists that subsumes this rule --> store it!
                        rules[size][antecedent_size][rule_string(antecedent, consequent)] = [antecedent_support, confidence]


# define names of output files
summary_output_file = os.path.join(args.output_folder, "summary-{0}-{1}.csv".format(args.support, args.confidence))
rules_output_template = os.path.join(args.output_folder, "rules-{0}-{1}.csv")

output_strings = ['rule_size,antecedent_size,num_rules\n']

# iterate over everything in order to create output
total_number_of_rules = 0
for rule_size, size_dict in rules.items():

    for antecedent_size, rule_instances in size_dict.items():
        
        number_of_rules = len(rule_instances.keys())
        # print length of list     
        if not args.quiet:
            print("\t\tNumber of rules with {0} concepts ({1} of them in antecedent): {2}".format(rule_size, antecedent_size, number_of_rules))
        total_number_of_rules += len(rule_instances.keys())
        
        # store for output into file
        output_strings.append(",".join([str(rule_size), str(antecedent_size), str(number_of_rules)]) + '\n')
        
        if len(rule_instances.keys()) > 0:
            # write resulting rules into file
            rules_file_name =  rules_output_template.format(rule_size, antecedent_size)
            with open(rules_file_name, 'w') as f:
                f.write("rule,support,confidence\n")
                for rule_string, rule_content in rule_instances.items():
                    line = "{0},{1},{2}\n".format(rule_string, rule_content[0], rule_content[1])
                    f.write(line)

print("\tSupport {0}, Confidence {1}, Total number of rules: {2}".format(args.support, args.confidence, total_number_of_rules))

with open(summary_output_file, 'w') as f:
    for line in output_strings:
        f.write(line)    
