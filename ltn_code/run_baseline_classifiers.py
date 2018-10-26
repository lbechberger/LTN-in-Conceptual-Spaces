# -*- coding: utf-8 -*-
"""
Runs some baseline classifiers for comparison reasons.

Created on Fri Oct 26 08:46:59 2018

@author: lbechberger
"""

import argparse
import util


# parse command line arguments
parser = argparse.ArgumentParser(description='knn in CS')
parser.add_argument('-q', '--quiet', action="store_true",
                    help = 'disables info output')                    
parser.add_argument('config_file', help = 'the config file to use')
parser.add_argument('config_name', help = 'the name of the configuration')
args = parser.parse_args()


# read config file
config = util.parse_config_file(args.config_file, args.config_name)


def get_predictions(baseline_type, concepts, data_set):
    
    predictions = {}

    label_frequencies = util.label_distribution(config["training_vectors"], config["concepts"])    
    
    for label in concepts:
        if baseline_type == 'constant':
            prediction = 0.5
        elif baseline_type == 'distribution':
            prediction = label_frequencies[label]
            
        predictions[label] = [prediction]*len(data_set)
      
    return predictions

for baseline_type in ['constant', 'distribution']:
    train_predictions = get_predictions(baseline_type, config["concepts"], config["training_vectors"])
    validation_predictions = get_predictions(baseline_type, config["concepts"], config["validation_vectors"])
    
    # evaluate the predictions
    eval_results = {}
    eval_results['contents'] = ['training', 'validation']
    eval_results['training'] = util.evaluate(train_predictions, config["training_vectors"], config["concepts"])
    eval_results['validation'] = util.evaluate(validation_predictions, config["validation_vectors"], config["concepts"])
    if not args.quiet:
        util.print_evaluation(eval_results)
    util.write_evaluation(eval_results, "output/{0}_{1}-baselines.csv".format(args.config_file.split('.')[0], args.config_name), "{0}_{1}".format(args.config_name, baseline_type))