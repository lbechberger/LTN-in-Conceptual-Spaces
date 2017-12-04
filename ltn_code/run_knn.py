# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 17:23:20 2017

@author: lbechberger
"""

import sys, random, ConfigParser
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

# read configuartion from the given config file
config = ConfigParser.RawConfigParser()
config.read(config_file_name)

# general setup
features_file = config.get(config_name, "features_file")
concepts_file = config.get(config_name, "concepts_file")
n_dims = config.getint(config_name, "num_dimensions")
training_percentage = config.getfloat(config_name, "training_percentage")

# parse features file
feature_vectors = []
with open(features_file, 'r') as f:
    for line in f:
        chunks = line.replace('\n','').replace('\r','').split(",")
        vec = map(float, chunks[:n_dims])
        labels = [label for label in chunks[n_dims:] if label != '']
        feature_vectors.append((labels, vec))
# shuffle them --> beginning of shuffle list will be treated as labeled, end as unlabeled
random.shuffle(feature_vectors)

# parse concepts file
concepts = []
with open(concepts_file, 'r') as f:
    for line in f:       
        label = line.replace('\n','').replace('\r', '')
        concepts.append(label)

# sample training_percentage of the data points as labeled ones
cutoff = int(len(feature_vectors) * training_percentage)
training_vectors = feature_vectors[:cutoff]
test_vectors = feature_vectors[cutoff:]

# re-format everything for scikit learn
training_data = np.array(list(map(lambda x: x[1], training_vectors)))
training_labels = np.array(list(map(lambda x: np.array(x[0]), training_vectors)))
training_labels = MultiLabelBinarizer(concepts).fit_transform(training_labels)
test_data = np.array(list(map(lambda x: x[1], test_vectors)))


# train and use kNN classifierqstat
classifier = KNeighborsClassifier(num_neighbors)
classifier.fit(training_data, training_labels)

predictions = {}
for label in concepts:
    predictions[label] = []

probabilities = classifier.predict_proba(test_data)
for i in range(len(probabilities)):
    predictions[concepts[i]] = map(lambda x: x[1] if len(x) > 1 else x, probabilities[i])

# evaluate on test set
print("One error on test data: {0}".format(util.one_error(predictions, test_vectors)))
print("Coverage on test data: {0}".format(util.coverage(predictions, test_vectors)))
