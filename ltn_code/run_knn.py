# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 17:23:20 2017

@author: lbechberger
"""

import sys, random
import util
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

# fix random seed to ensure reproducibility
random.seed(42)


# parse command line arguments
if sys.argv < 4:
    raise Exception("Need two arguments: 'python run_ltn.py config.cfg config_name k'")
config_file_name = sys.argv[1]
config_name = sys.argv[2]
num_neighbors = int(sys.argv[3])

# read config file
config = util.parse_config_file(config_file_name, config_name)

# re-format everything for scikit learn
training_data = np.array(list(map(lambda x: x[1], config["training_vectors"])))
training_labels = np.array(list(map(lambda x: np.array(x[0]), config["training_vectors"])))
training_labels = MultiLabelBinarizer(config["concepts"]).fit_transform(training_labels)
validation_data = np.array(list(map(lambda x: x[1], config["validation_vectors"])))

# train and use kNN classifier
classifier = KNeighborsClassifier(num_neighbors)
classifier.fit(training_data, training_labels)

def get_predictions(classifier, concepts, data):
    predictions = {}
    for label in concepts:
        predictions[label] = []
    
    probabilities = classifier.predict_proba(data)
    
    for i in range(len(probabilities)):
        predictions[concepts[i]] = map(lambda x: x[1] if len(x) > 1 else x, probabilities[i])

    return predictions

train_predictions = get_predictions(classifier, config["concepts"], training_data)
validation_predictions = get_predictions(classifier, config["concepts"], validation_data)

# evaluate the predictions
util.evaluate(train_predictions, config["training_vectors"], validation_predictions, config["validation_vectors"])