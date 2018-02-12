# -*- coding: utf-8 -*-
"""
Some utility functions (like "one error" evaluation).

Created on Mon Dec  4 12:19:52 2017

@author: lbechberger
"""
from configparser import RawConfigParser
from math import log
import os, fcntl

def parse_config_file(config_file_name, config_name):
    """Extracts all parameters of interest form the given config file."""
    result = {}
    config = RawConfigParser()
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
            vec = list(map(float, chunks[:n_dims]))
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
        for label, memberships in predictions.items():
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
        for label, memberships in predictions.items():
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
        for label, memberships in predictions.items():
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

def exact_match_prefix(predictions, vectors):
    """Computes the exact match for the given vectors and the given predictions by looking at the highest ranked predictions.
    
    If there are n ground truth labels, check whether the first n predictions are correct. The higher, the better."""
    
    count = 0
    idx = 0
    for (true_labels, vector) in vectors:
        n_labels = len(true_labels)
        filtered_predictions = []
        for label, memberships in predictions.items():
            filtered_predictions.append((label, memberships[idx]))
        filtered_predictions.sort(key = lambda x: x[1], reverse = True) # sort in descending order based on membership
        filtered_predictions = list(map(lambda x: x[0], filtered_predictions))[:n_labels]
        
        equivalent = True
        for pred in filtered_predictions:
            if pred not in true_labels:
                equivalent = False
                break
        if equivalent:
            count += 1
        idx += 1
    
    exact_match_prefix = (1.0 * count) / len(vectors)
    return exact_match_prefix

def cross_entropy_loss(predictions, vectors, all_labels):
    """Computes the cross entropy loss between the predictions and the ground truth labels.
    
    How close are the numeric scores to the values of 0 and 1? The lower, the better."""
    
    sum_of_cross_entropies = 0
    idx = 0
    for (true_labels, vector) in vectors:
        
        for label in true_labels:
            if predictions[label][idx] == 0.0:
                sum_of_cross_entropies -= 1000
            else:
                sum_of_cross_entropies += log(predictions[label][idx], 2)
        
        false_labels = [label for label in all_labels if label not in true_labels]
        for label in false_labels:
            if predictions[label][idx] == 1.0:
                sum_of_cross_entropies -= 1000
            else:
                sum_of_cross_entropies += log(1.0 - predictions[label][idx], 2)
        
        idx += 1
    
    return (-1.0 * sum_of_cross_entropies) / len(vectors)

def label_wise_hit_rate(predictions, vectors, all_labels):
    """Computes for each label the percentage of times where it was ranked before the first invalid label.

    Looks only at cases where the label was in the ground truth. The higher, the better."""   
    
    appearances = {}
    hits = {}
    for label in all_labels:
        appearances[label] = 0.0
        hits[label] = 0.0
    idx = 0
    for (true_labels, vector) in vectors:
        filtered_predictions = []
        for label, memberships in predictions.items():
            filtered_predictions.append((label, memberships[idx]))
        filtered_predictions.sort(key = lambda x: x[1], reverse = True) # sort in descending order based on membership
        for label in true_labels:
            appearances[label] += 1
        for (label, membership) in filtered_predictions:
            if label in true_labels:
                hits[label] += 1
            else:
                break
        idx += 1
        
    result = {}
    minimal_result = float("inf")
    summed_result = 0
    counter = 0
    for label in all_labels:
        if appearances[label] == 0:
            result[label] = None
        else:
            result[label] = (1.0 * hits[label]) / appearances[label]
            minimal_result = min(minimal_result, result[label])
            summed_result += result[label]
            counter += 1

    result['min'] = minimal_result
    result['mean'] = (1.0 * summed_result) / counter
    result['contents'] = ['min', 'mean'] + sorted(all_labels)
        
    return result

def evaluate(predictions, vectors, all_labels):
    """Evaluate the predictions on the given data set."""
    
    result = {}
    result['one_error'] = one_error(predictions, vectors)
    result['coverage'] = coverage(predictions, vectors)
    result['ranking_loss'] = ranking_loss(predictions, vectors, all_labels)
    result['average_precision'] = average_precision(predictions, vectors)
    result['exact_match_prefix'] = exact_match_prefix(predictions, vectors)
    result['cross_entropy_loss'] = cross_entropy_loss(predictions, vectors, all_labels)
    result['label_wise_hit_rate'] = label_wise_hit_rate(predictions, vectors, all_labels)
    result['contents'] = ['one_error', 'coverage', 'ranking_loss', 'average_precision', 'exact_match_prefix', 'cross_entropy_loss', 'label_wise_hit_rate']
    
    return result

def print_evaluation(evaluation_results):
    """Print the evaluation results on all data sets."""
    
    for data_set in evaluation_results['contents']:
        
        evaluation = evaluation_results[data_set]
        print("\n{0}:".format(data_set))
        print("-"*(len(data_set) + 1))
        for name in evaluation['contents']:
            if name == "label_wise_hit_rate":
                print("label-wise hit rate:")
                for label in evaluation[name]['contents']:
                    print("{0}: {1}".format(label, evaluation[name][label]))
            else:
                print("{0}: {1}".format(name, evaluation_results[data_set][name]))
            
def write_evaluation(evaluation_results, file_name, config_name):
    """Write the evaluation results into a csv file."""

    # write headline if necessary
    if not os.path.exists(file_name):
        with open(file_name, 'w') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write("config,data_set,")
            data_set = evaluation_results['contents'][0]
            evaluation = evaluation_results[data_set]
            for name in evaluation['contents']:
                if name=="label_wise_hit_rate":
                    for label in evaluation[name]['contents']:
                        f.write("{0},".format(label))
                else:
                    f.write("{0},".format(name))
            f.write("\n")
            fcntl.flock(f, fcntl.LOCK_UN)
    
    # write content
    with open(file_name, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        for data_set in evaluation_results['contents']:
            evaluation = evaluation_results[data_set]
            line = "{0},{1},".format(config_name, data_set)
            for name in evaluation['contents']:
                if name=="label_wise_hit_rate":
                    for label in evaluation[name]['contents']:
                        line += "{0},".format(evaluation[name][label])
                else:
                    line += "{0},".format(evaluation[name])
            line += "\n"
            f.write(line)
        fcntl.flock(f, fcntl.LOCK_UN)


def distinct_label_set(vectors):
    """Computes the distinct label set of the given data set.
    
    How many distinct label combinations are there in the data set?"""

    label_sets = []   
    for (true_labels, vector) in vectors:
        label_set = str(set(true_labels))
        label_sets.append(label_set)
    
    return len(set(label_sets))

def proportion_of_distinct_label_set(vectors):
    """Computes the proportion of distinct label set of the given data set.
    
    Distinct label set normalized by the total number of training instances."""
    
    return (1.0 * distinct_label_set(vectors)) / len(vectors)

def label_cardinality(vectors):
    """Computes the label cardinality of the given data set.
    
    How many labels per example do we have on average?"""
    
    total_number_of_labels = 0
    for (true_labels, vector) in vectors:
        total_number_of_labels += len(true_labels)
    
    return (1.0 * total_number_of_labels) / len(vectors)

def label_density(vectors, all_labels):
    """Computes the label density of the given data set.
    
    Label cardinality normalized by the total number of labels."""
    
    return (1.0 * label_cardinality(vectors)) / len(all_labels)

def label_distribution(vectors, all_labels):
    """Computes the distribution of labels for the given data set.
    
    How often do the labels occur percentage-wise in the data set?"""
    
    label_counter = {}
    for label in all_labels:
        label_counter[label] = 0
    
    for (true_labels, vector) in vectors:
        for label in true_labels:
            label_counter[label] += 1
    
    result = {}
    for label in all_labels:
        result[label] = (1.0 * label_counter[label]) / len(vectors)
    
    return result

def data_set_characteristics(train_vectors, validation_vectors, test_vectors, all_labels):
    print("\nTraining Data:")
    print("--------------")
    print("Size: {0}".format(len(train_vectors)))
    print("Distinct label set: {0}".format(distinct_label_set(train_vectors)))
    print("Proportion of distinct label set: {0}".format(proportion_of_distinct_label_set(train_vectors)))
    print("Label cardinality: {0}".format(label_cardinality(train_vectors)))
    print("Label density: {0}".format(label_density(train_vectors, all_labels)))
    print("Label distribution: {0}".format(label_distribution(train_vectors, all_labels)))
    
    print("\nValidation Data:")
    print("----------------")
    print("Size: {0}".format(len(validation_vectors)))
    print("Distinct label set: {0}".format(distinct_label_set(validation_vectors)))
    print("Proportion of distinct label set: {0}".format(proportion_of_distinct_label_set(validation_vectors)))
    print("Label cardinality: {0}".format(label_cardinality(validation_vectors)))
    print("Label density: {0}".format(label_density(validation_vectors, all_labels)))
    print("Label distribution: {0}".format(label_distribution(validation_vectors, all_labels)))
    
    print("\nTest Data:")
    print("----------")
    print("Size: {0}".format(len(test_vectors)))
    print("Distinct label set: {0}".format(distinct_label_set(test_vectors)))
    print("Proportion of distinct label set: {0}".format(proportion_of_distinct_label_set(test_vectors)))
    print("Label cardinality: {0}".format(label_cardinality(test_vectors)))
    print("Label density: {0}".format(label_density(test_vectors, all_labels)))
    print("Label distribution: {0}".format(label_distribution(test_vectors, all_labels)))
    