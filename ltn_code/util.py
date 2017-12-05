# -*- coding: utf-8 -*-
"""
Some utility functions (like "one error" evaluation).

Created on Mon Dec  4 12:19:52 2017

@author: lbechberger
"""
import ConfigParser, random

def parse_config_file(config_file_name, config_name):
    """Extracts all parameters of interest form the given config file."""
    result = {}
    config = ConfigParser.RawConfigParser()
    config.read(config_file_name)
    
    # general setup
    result["features_file"] = config.get(config_name, "features_file")
    result["concepts_file"] = config.get(config_name, "concepts_file")
    result["rules_file"] = config.get(config_name, "rules_file")
    result["num_dimensions"] = config.getint(config_name, "num_dimensions")
    result["training_percentage"] = config.getfloat(config_name, "training_percentage")
    result["max_iter"] = config.getint(config_name, "max_iter")
    
    # LTN setup
    def read_ltn_variable(name, is_int = False, is_float = False):
        buf = None        
        if config.has_option(config_name, name):
            buf = config.get(config_name, name)
        elif config.has_option("ltn-default", name):
            buf = config.get("ltn-default", name)  
        
        if buf != None:
            if is_int:
                result[name] = int(buf)
            elif is_float:
                result[name] = float(buf)
            else:
                result[name] = buf
        
    read_ltn_variable("ltn_layers", is_int = True)
    read_ltn_variable("ltn_smooth_factor", is_float = True)     
    read_ltn_variable("ltn_tnorm")
    read_ltn_variable("ltn_aggregator")        
    read_ltn_variable("ltn_optimizer")
    read_ltn_variable("ltn_clauses_aggregator")
    read_ltn_variable("ltn_positive_fact_penalty", is_float = True)
    read_ltn_variable("ltn_norm_of_u", is_float = True)

    train_vecs, test_vecs = parse_features_file(result["features_file"], result["training_percentage"], result["num_dimensions"])
    result["training_vectors"] = train_vecs
    result["test_vectors"] = test_vecs

    result["concepts"] = parse_concepts_file(result["concepts_file"])

    return result

def parse_features_file(file_name, training_percentage, n_dims):
    feature_vectors = []
    with open(file_name, 'r') as f:
        for line in f:
            chunks = line.replace('\n','').replace('\r','').split(",")
            vec = map(float, chunks[:n_dims])
            labels = [label for label in chunks[n_dims:] if label != '']
            feature_vectors.append((labels, vec))
    # shuffle them --> beginning of shuffle list will be treated as labeled, end as unlabeled
    random.shuffle(feature_vectors)

    # sample training_percentage of the data points as labeled ones
    cutoff = int(len(feature_vectors) * training_percentage)
    training_vectors = feature_vectors[:cutoff]
    test_vectors = feature_vectors[cutoff:]
    
    return training_vectors, test_vectors

def parse_concepts_file(file_name):
    concepts = []
    with open(file_name, 'r') as f:
        for line in f:       
            concepts.append(line.replace('\n','').replace('\r', ''))
    return concepts


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
