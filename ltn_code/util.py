# -*- coding: utf-8 -*-
"""
Some utility functions (like one error evaluation).

Created on Mon Dec  4 12:19:52 2017

@author: lbechberger
"""

def one_error(predictions, vectors):
    """Computes the one error for the given vectors and the given predictions."""
    idx = 0
    num_incorrect = 0
    for (true_labels, vector) in vectors:
        predicted_label = None
        predicted_confidence = 0
        for label, memberships in predictions.iteritems():
            conf = memberships[idx]
            if conf > predicted_confidence:
                predicted_confidence = conf
                predicted_label = label
        if predicted_label not in true_labels:
            num_incorrect += 1
        idx += 1
    
    one_error = (1.0 * num_incorrect) / len(vectors)
    return one_error

def coverage(predictions, vectors):
    """Computes the coverage for the given vectors and the given predictions."""
    idx = 0
    summed_depth = 0
    for (true_labels, vector) in vectors:
        filtered_predictions = []
        for label, memberships in predictions.iteritems():
            filtered_predictions.append((label, memberships[idx]))
        filtered_predictions.sort(key = lambda x: x[1], reverse = True) # sort in descending order based on membership
        depth = 0
        labels_to_find = list(true_labels)
        while depth < len(filtered_predictions) and len(labels_to_find) > 0:
            if filtered_predictions[depth][0] in labels_to_find:
                labels_to_find.remove(filtered_predictions[depth][0])
            depth += 1
        summed_depth += depth
        idx += 1
    
    coverage = (1.0 * summed_depth) / len(vectors)
    return coverage

def evaluate(train_predictions, train_vectors, test_predictions, test_vectors):
    """Evaluate the predictions both on the training and the test set."""
    
    # training data (to get an idea about overfitting)
    print("One error on training data: {0}".format(one_error(train_predictions, train_vectors)))
    print("Coverage on training data: {0}".format(coverage(train_predictions, train_vectors)))
    
    # test data (the stuff that matters)
    print("One error on test data: {0}".format(one_error(test_predictions, test_vectors)))
    print("Coverage on test data: {0}".format(coverage(test_predictions, test_vectors)))
