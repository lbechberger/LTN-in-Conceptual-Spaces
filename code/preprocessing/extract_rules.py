# -*- coding: utf-8 -*-
"""
Extracts rules from data set based on the label counting results.

Created on Mon Feb 11 15:04:57 2019

@author: lbechberger
"""

import argparse, pickle, os

# parse command line arguments
parser = argparse.ArgumentParser(description='rule extraction based on label counting')                  
parser.add_argument('input_file', help = 'path to the input pickle file')
parser.add_argument('-o', '--output_folder', help = 'path to output folder for the rule files', default = '.')                
parser.add_argument('-s', '--support', type = float, help = 'threshold for rule support', default = 0.01)
parser.add_argument('-i', '--improvement', type = float, help = 'threshold for rule improvement', default = 0.1)  
parser.add_argument('-c', '--confidence', type = float, help = 'threshold for rule confidence', default = 0.9)                
parser.add_argument('-q', '--quiet', action="store_true", help = 'disables info output')                    
args = parser.parse_args()

if not args.quiet:
    print("\tLoading data ...")
# import data
data = pickle.load(open(args.input_file, 'rb'))
rule_types = data['rule_types']
rule_strings = data['rule_strings']

rules = {}
for rule_type in rule_types:
    if not args.quiet:
        print("\t... {0}".format(rule_type))

    file_name = args.input_file.replace('central', rule_type)
    rules[rule_type] = pickle.load(open(file_name, 'rb'))

epsilon = 1e-10

# return confidence if exists, otherwise zero
def get_confidence(rule_type, rule_string):
    if rule_string in rules[rule_type]:
        return rules[rule_type][rule_string][1]
    return 0

# filter for rule improvements
def filter_improvement(complex_rule_type, simple_rule_type_one, simple_rule_type_two, filter_type):

    if not args.quiet:
        print("\t... {0}".format(complex_rule_type))

    for rule, values in rules[complex_rule_type].copy().items():
        
        complex_confidence = values[1]
        
        if filter_type == 'and':
            first_confidence = get_confidence(simple_rule_type_one, "_".join([values[2], values[4]]))
            second_confidence = get_confidence(simple_rule_type_two, "_".join([values[3], values[4]]))
        elif filter_type == 'or':
            first_confidence = get_confidence(simple_rule_type_one, "_".join([values[2], values[3]]))
            second_confidence = get_confidence(simple_rule_type_two, "_".join([values[2], values[4]]))
        else:
            raise Exception('invalid filter type')
        
        first_improvement = (complex_confidence + epsilon) / (first_confidence + epsilon)
        second_improvement = (complex_confidence + epsilon) / (second_confidence + epsilon)
        improvement = min(first_improvement, second_improvement)
        
        if improvement - 1 < args.improvement:
            del rules[complex_rule_type][rule]

# filter out rules with insufficient support
if not args.quiet:
    print("\tFiltering all rules for support ...")
for rule_type in rule_types:
    for rule, values in rules[rule_type].copy().items():
        if values[0] < args.support:
            del rules[rule_type][rule]

# filter out rules with insufficient confidence
if not args.quiet:
    print("\tFiltering all rules for confidence ...")
for rule_type in rule_types:
    for rule, values in rules[rule_type].copy().items():
        if values[1] < args.confidence:
            del rules[rule_type][rule]

# filter out rules with poor improvement
if not args.quiet:
    print("\tFiltering AND rules for improvement ...")

filter_improvement('pANDpIMPLp', 'pIMPLp', 'pIMPLp', 'and')
filter_improvement('pANDnIMPLp', 'pIMPLp', 'nIMPLp', 'and')
filter_improvement('nANDnIMPLp', 'nIMPLp', 'nIMPLp', 'and')

if not args.quiet:
    print("\tFiltering OR rules for improvement ...")
filter_improvement('pIMPLpORp', 'pIMPLp', 'pIMPLp', 'or')
filter_improvement('nIMPLpORp', 'nIMPLp', 'nIMPLp', 'or')


# define names of output files
summary_output_file = os.path.join(args.output_folder, "rules-{0}-{1}-{2}.csv".format(args.support, args.improvement, args.confidence))
rules_output_template = os.path.join(args.output_folder, "{0}.csv")

output_strings = ['rule_type,num_rules\n']

if not args.quiet:
    print("\tEvaluating the rules ...")

for rule_type, rule_instances in rules.items():

    # print length of list     
    if not args.quiet:
        print("\t\tNumber of rules of type {0} left for support {1}, confidence {2}, improvement {3}: {4} ".format(rule_type, args.support, args.confidence, args.improvement,
              len(rule_instances.keys())))
    
    # store for output into file
    output_strings.append(",".join([rule_type, str(len(rule_instances.keys()))]) + '\n')
    
    if len(rule_instances.keys()) > 0:
        # write resulting rules into file
        rules_file_name =  rules_output_template.format(rule_type)
        with open(rules_file_name, 'w') as f:
            f.write("rule,support,confidence\n")
            for rule_string, rule_content in rule_instances.items():
                if len(rule_content) == 4:
                    output_rule_string = rule_strings[rule_type].format(rule_content[2], rule_content[3])
                elif len(rule_content) == 5:
                    output_rule_string = rule_strings[rule_type].format(rule_content[2], rule_content[3], rule_content[4])
                else:
                    raise(Exception("invalid length of rule information"))
                line = "{0},{1},{2}\n".format(output_rule_string, rule_content[0], rule_content[1])
                f.write(line)
    
    if not args.quiet:
        print("")

with open(summary_output_file, 'w') as f:
    for line in output_strings:
        f.write(line)    

if not args.quiet:
    print("DONE")
