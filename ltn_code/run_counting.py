# -*- coding: utf-8 -*-
"""
Runs the label counting algorithm in order to extract rules.

Created on Thu Oct 18 11:03:27 2018

@author: lbechberger
"""

import random, argparse
import util

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

# first look for simple rules like "first_concept implies second_concept"
for first_concept in config["concepts"]:
    for second_concept in config["concepts"]:
        if first_concept == second_concept:
            continue
        count_is_first = 0
        count_is_both = 0
        count_is_not_first = 0
        count_is_only_second = 0
        for (labels, vector) in config["training_vectors"]:
            if first_concept in labels:
                count_is_first += 1
                if second_concept in labels:
                    count_is_both += 1
            else:
                count_is_not_first += 1
                if second_concept in labels:
                    count_is_only_second += 1
        pos_pos_implication = count_is_both / count_is_first
        pos_neg_implication = 1 - pos_pos_implication
        neg_pos_implication = count_is_only_second / count_is_not_first
        neg_neg_implication = 1 - neg_pos_implication
        
        print("{0} and {1}: {2}, {3}, {4}, {5}".format(first_concept, second_concept, pos_pos_implication, pos_neg_implication, neg_pos_implication, neg_neg_implication))