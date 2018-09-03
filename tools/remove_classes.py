# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 09:32:14 2018

Removes the given list of classes from the data set.

@author: lbechberger
"""

import sys, argparse, os
from configparser import ConfigParser
sys.path.append("ltn_code/")
import util

# parse command line arguments
parser = argparse.ArgumentParser(description='remove classes')
parser.add_argument('-q', '--quiet', action="store_true",
                    help = 'disables info output')                    
parser.add_argument('config_file', help = 'the config file to use')
parser.add_argument('config_name', help = 'the name of the configuration to use as a starting point')
parser.add_argument('to_remove', help = 'list of all the classes to remove (one string, class names separated by commas)')
parser.add_argument('--output_suffix', default = "cleaned", help = 'the suffix to append to all the names for the new configuration')
args = parser.parse_args()

# read config file
config = util.parse_config_file(args.config_file, args.config_name)

unwanted_classes = args.to_remove.split(",")
file_suffix = "_{0}.".format(args.output_suffix)
folder_suffix = "_{0}/".format(args.output_suffix)
config_suffix = "_{0}".format(args.output_suffix)

# first update the concepts_file (simply remove the unwanted classes)
new_concepts_file = config['concepts_file'].replace('.', file_suffix)
with open(config['concepts_file'], 'r') as input_file:
    with open(new_concepts_file, 'w') as output_file:
        for line in input_file:
            clean_line = line.replace('\n', '')
            if clean_line not in unwanted_classes:
                output_file.write(line)


# now update the features files 
# (remove the class labels of the unwanted classes from all feature vectors; if vector is left without a label: delete it)
new_features_folder = config['features_folder'][:-1] + folder_suffix
if not os.path.exists(new_features_folder):
    os.makedirs(new_features_folder)
for part in ['training', 'validation', 'test']:
    feature_vectors = config['{0}_vectors'.format(part)]
    with open("{0}{1}.csv".format(new_features_folder, part), 'w') as output_file:
        for vector in feature_vectors:
            labels = vector[0]
            features = vector[1]
            reduced_labels = [label for label in labels if label not in unwanted_classes]
            if len(reduced_labels) > 0:
                output_file.write("{0},{1}\n".format(','.join(map(lambda x: str(x), features)), ','.join(reduced_labels)))


# now update the rules file (remove all rules that contain one of the unwanted classes)
new_rules_file = config['rules_file'].replace('.', file_suffix)
with open(config['rules_file'], 'r') as input_file:
    with open(new_rules_file, 'w') as output_file:
        for line in input_file:
            is_okay = True
            for unwanted_class in unwanted_classes:
                if unwanted_class in line:
                    is_okay = False
            if is_okay:
                output_file.write(line)


# finally: create a new configuration in the config file
new_config_name = args.config_name + config_suffix
config_parser = ConfigParser()
config_parser.read(args.config_file)
new_config = dict(config_parser[args.config_name])

new_config["concepts_file"] = new_concepts_file
new_config["features_folder"] = new_features_folder
new_config["rules_file"] = new_rules_file

config_parser[new_config_name] = new_config
with open(args.config_file, 'w') as f:
    config_parser.write(f)