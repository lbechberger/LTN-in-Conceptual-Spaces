# -*- coding: utf-8 -*-
"""
Some utility functions (like "one error" evaluation).

Created on Mon Dec  4 12:19:52 2017

@author: lbechberger
"""
import ConfigParser

def parse_config_file(config_file_name, config_name):
    """Extracts all parameters of interest form the given config file."""
    result = {}
    config = ConfigParser.RawConfigParser()
    config.read(config_file_name)
    
    # general setup
    result["features_folder"] = config.get(config_name, "features_folder")
    result["concepts_file"] = config.get(config_name, "concepts_file")
    result["rules_file"] = config.get(config_name, "rules_file")
    result["num_dimensions"] = config.getint(config_name, "num_dimensions")
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
    read_ltn_variable("ltn_epsilon", is_float = True)

    for part in ["training", "validation", "test"]:
        result["{0}_vectors".format(part)] = parse_features_file("{0}{1}.csv".format(result["features_folder"], part), result["num_dimensions"])

    result["concepts"] = parse_concepts_file(result["concepts_file"])

    return result

def parse_features_file(file_name, n_dims):
    feature_vectors = []
    with open(file_name, 'r') as f:
        for line in f:
            chunks = line.replace('\n','').replace('\r','').split(",")
            vec = map(float, chunks[:n_dims])
            labels = [label for label in chunks[n_dims:] if label != '']
            feature_vectors.append((labels, vec))
    
    return feature_vectors


def parse_concepts_file(file_name):
    concepts = []
    with open(file_name, 'r') as f:
        for line in f:       
            concepts.append(line.replace('\n','').replace('\r', ''))
    return concepts


def one_error(predictions, vectors):
    """Computes the one error for the given vectors and the given predictions. 

    How often is the highest-ranked prediction not in the ground-truth labels? The smaller, the better."""
    
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
    """Computes the coverage for the given vectors and the given predictions. 
    
    How far do you need to go down the ordered list of predictions to find all labels? The smaller, the better."""
    
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

def ranking_loss(predictions, vectors, all_labels):
    """Computes the ranking loss for the given vectors and the given predictions. 
    
    How many label pairs are incorrectly ordered? The smaller, the better."""
    
    idx = 0
    summed_loss = 0
    count = 0
    for (true_labels, vector) in vectors:
        local_loss = 0
        false_labels = [label for label in all_labels if label not in true_labels]
        
        for true_label in true_labels:
            for false_label in false_labels:
                if predictions[true_label][idx] < predictions[false_label][idx]:
                    local_loss += 1
        if len(true_labels) > 0 and len(false_labels) > 0:
            summed_loss += (1.0 * local_loss) / (len(true_labels) * len(false_labels))
            count += 1
        idx += 1
        
    ranking_loss = (1.0 * summed_loss) / count
    return ranking_loss

def average_precision(predictions, vectors):
    """Computes the average precision for the given vectors and the given predictions. 
    
    How many relevant labels are ranked before each relevant label? The higher, the better."""
    
    idx = 0
    summed_precision = 0
    count = 0
    for (true_labels, vector) in vectors:
        filtered_predictions = []
        for label, memberships in predictions.iteritems():
            filtered_predictions.append((label, memberships[idx]))
        filtered_predictions.sort(key = lambda x: x[1], reverse = True) # sort in descending order based on membership
        filtered_predictions = list(map(lambda x: x[0], filtered_predictions))
        precision = 0
        
        for true_label in true_labels:
            rank = filtered_predictions.index(true_label)
            if rank == 0:
                precision += 1
            else:
                true_labels_before = len([label for label in true_labels if label in filtered_predictions[:rank]])
                precision += (1.0 * true_labels_before) / rank
            
        
        if len(true_labels) > 0:
            summed_precision += (1.0 * precision) / len(true_labels)
            count += 1
            
        idx += 1    
    
    average_precision = (1.0 * summed_precision) / count
    return average_precision
    
def evaluate(train_predictions, train_vectors, validation_predictions, validation_vectors, all_labels):
    """Evaluate the predictions both on the training and the validation set."""
    
    # training data (to get an idea about overfitting)
    print(" ")
    print("One error on training data: {0}".format(one_error(train_predictions, train_vectors)))
    print("Coverage on training data: {0}".format(coverage(train_predictions, train_vectors)))
    print("Ranking loss on training data: {0}".format(ranking_loss(train_predictions, train_vectors, all_labels)))
    print("Average precision on training data: {0}".format(average_precision(train_predictions, train_vectors)))
    
    # test data (the stuff that matters)
    print(" ")
    print("One error on validation data: {0}".format(one_error(validation_predictions, validation_vectors)))
    print("Coverage on validation data: {0}".format(coverage(validation_predictions, validation_vectors)))
    print("Ranking loss on validation data: {0}".format(ranking_loss(validation_predictions, validation_vectors, all_labels)))
    print("Average precision on validation data: {0}".format(average_precision(validation_predictions, validation_vectors)))
    