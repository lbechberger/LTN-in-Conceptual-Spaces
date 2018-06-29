# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 17:23:20 2017

@author: lbechberger
"""

import random, argparse
import util
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

# fix random seed to ensure reproducibility
random.seed(42)

# parse command line arguments
parser = argparse.ArgumentParser(description='knn in CS')
parser.add_argument('-q', '--quiet', action="store_true",
                    help = 'disables info output')                    
parser.add_argument('config_file', help = 'the config file to use')
parser.add_argument('config_name', help = 'the name of the configuration')
parser.add_argument('num_neighbors', type = int, help = 'which k to use for knn')
args = parser.parse_args()


# read config file
config = util.parse_config_file(args.config_file, args.config_name)

# re-format everything for scikit learn
training_data = np.array(list(map(lambda x: x[1], config["training_vectors"])))
training_labels = np.array(list(map(lambda x: np.array(x[0]), config["training_vectors"])))
training_labels = MultiLabelBinarizer(config["concepts"]).fit_transform(training_labels)
validation_data = np.array(list(map(lambda x: x[1], config["validation_vectors"])))

# train and use kNN classifier
classifier = KNeighborsClassifier(args.num_neighbors)
classifier.fit(training_data, training_labels)

def get_predictions(classifier, concepts, data):
    predictions = {}
    for label in concepts:
        predictions[label] = []
    
    probabilities = classifier.predict_proba(data)
    
    for i in range(len(probabilities)):
        predictions[concepts[i]] = list(map(lambda x: x[1] if len(x) > 1 else x, probabilities[i]))

    return predictions

train_predictions = get_predictions(classifier, config["concepts"], training_data)
validation_predictions = get_predictions(classifier, config["concepts"], validation_data)

# evaluate the predictions
eval_results = {}
eval_results['contents'] = ['training', 'validation']
eval_results['training'] = util.evaluate(train_predictions, config["training_vectors"], config["concepts"])
eval_results['validation'] = util.evaluate(validation_predictions, config["validation_vectors"], config["concepts"])
if not args.quiet:
    util.print_evaluation(eval_results)
util.write_evaluation(eval_results, "output/{0}_{1}-knn.csv".format(args.config_file.split('.')[0], args.config_name), "{0}_k{1}".format(args.config_name, args.num_neighbors))
