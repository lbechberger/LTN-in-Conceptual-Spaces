# -*- coding: utf-8 -*-
"""
Some utility functions (like "one error" evaluation).

Created on Mon Dec  4 12:19:52 2017

@author: lbechberger
"""
from configparser import RawConfigParser
from math import log
import os, fcntl
import ast
import itertools

###############
# INPUT FILES #
###############
#------------------------------------------------------------------------------

def parse_config_file(config_file_name, config_name):
    """Extracts all parameters of interest form the given config file."""
    result = {}
    config = RawConfigParser()
    config.read(config_file_name)
    
    # general setup
    def parse_range(key):
        value = result[key]
        parsed_value = ast.literal_eval(value)
        if isinstance(parsed_value, list):
            result[key] = parsed_value
        else:
            result[key] = [parsed_value]

    result["features_folder"] = config.get(config_name, "features_folder")
    result["concepts_file"] = config.get(config_name, "concepts_file")
    result["rules_file"] = config.get(config_name, "rules_file")
    result["num_dimensions"] = config.getint(config_name, "num_dimensions")
    result["epochs"] = config.get(config_name, "epochs")
    parse_range('epochs')    

    
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
        
    read_ltn_variable("layers", is_int = True)
    read_ltn_variable("smooth_factor", is_float = True)     
    read_ltn_variable("tnorm")
    read_ltn_variable("aggregator")        
    read_ltn_variable("optimizer")
    read_ltn_variable("clauses_aggregator")
    read_ltn_variable("positive_fact_penalty", is_float = True)
    read_ltn_variable("norm_of_u", is_float = True)
    read_ltn_variable("epsilon", is_float = True)

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

##############
# EVALUATION #
##############
#------------------------------------------------------------------------------

def get_sorted_grouped_predictions(predictions, idx):
    """Helper function for getting all predictions for the example with the given index.
    
    Returns a list of [list of labels, confidence] pairs that is sorted in descending order based on the confidence."""

    # group the labels together based on their predicted confidence
    prediction_groups = {}
    for label, memberships in predictions.items():
        prediction = memberships[idx]
        
        if prediction in prediction_groups:
            prediction_groups[prediction].append(label)
        else:
            prediction_groups[prediction] = [label]

    # put into a list and sort in descending order based on membership
    grouped_predictions = []    
    for prediction, labels in prediction_groups.items():
        grouped_predictions.append([labels, prediction])
    grouped_predictions.sort(key = lambda x: x[1], reverse = True)
    
    return grouped_predictions

def one_error(predictions, vectors):
    """Computes the one error for the given vectors and the given predictions. 

    How often is the highest-ranked prediction not in the ground-truth labels? The smaller, the better."""
    
    idx = 0
    sum_of_local_one_errors = 0
    for (true_labels, vector) in vectors:
        predicted_labels = []
        predicted_confidence = 0
        for label, memberships in predictions.items():
            conf = memberships[idx]
            
            if conf > predicted_confidence:
                # higher confidence than before: replace
                predicted_confidence = conf
                predicted_labels = [label]
            elif conf == predicted_confidence:
                # exact same confidence as before: add to list
                predicted_labels.append(label)
        
        # what fraction of the highest ranked classes belongs to the ground truth?
        local_one_error = 0
        for predicted_label in predicted_labels:
            if predicted_label not in true_labels:
                local_one_error += 1
        local_one_error /= len(predicted_labels)
        
        sum_of_local_one_errors += local_one_error
        idx += 1
    
    # average over all training examples
    one_error = sum_of_local_one_errors / len(vectors)
    return one_error

def coverage(predictions, vectors):
    """Computes the coverage for the given vectors and the given predictions. 
    
    How far do you need to go down the ordered list of predictions to find all labels? The smaller, the better."""
    
    idx = 0
    summed_depth = 0
    for (true_labels, vector) in vectors:
        
        grouped_predictions = get_sorted_grouped_predictions(predictions, idx)        
        
        depth = 0
        inner_idx = 0
        labels_to_find = list(true_labels)
        while inner_idx < len(grouped_predictions) and len(labels_to_find) > 0:
            labels_to_find = [element for element in labels_to_find if element not in grouped_predictions[inner_idx][0]]
            depth += len(grouped_predictions[inner_idx][0])
            inner_idx += 1
            
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

        grouped_predictions = get_sorted_grouped_predictions(predictions, idx)        
                
        precision = 0
        
        for true_label in true_labels:
            index_where_found = 0
            true_labels_before = 0
            rank = 0
            while index_where_found < len(grouped_predictions):
                current_labels = grouped_predictions[index_where_found][0]
                true_labels_before += len([label for label in true_labels if label in current_labels])
                rank += len(current_labels)
                
                if true_label in current_labels:
                    break
                index_where_found += 1
                
            
            precision += true_labels_before / rank            
        
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


#######################
# DATA SET PROPERTIES #
#######################
#------------------------------------------------------------------------------

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


###################
# RULE EXTRACTION #
###################
#------------------------------------------------------------------------------

# rule types we're interested in. "p" means "positive", "n" means "negative", 
#"IMPL" means "implies", and "DIFF" means "is different from"
rule_types = ['pIMPLp', 'pIMPLn', 'nIMPLp', 'nIMPLn',
              'pANDpIMPLp', 'pANDpIMPLn', 'pANDnIMPLp', 'pANDnIMPLn', 'nANDpIMPLp', 'nANDpIMPLn', 'nANDnIMPLp', 'nANDnIMPLn',
              'pIMPLpORp', 'pIMPLpORn', 'pIMPLnORp', 'pIMPLnORn', 'nIMPLpORp', 'nIMPLpORn', 'nIMPLnORp', 'nIMPLnORn',
              'pDIFFp']

# dicitionary mapping rule types to desired output string
rule_strings = {'pIMPLp' : '{0} IMPLIES {1}', 'pIMPLn' : '{0} IMPLIES (NOT {1})', 
                'nIMPLp' : '(NOT {0}) IMPLIES {1}', 'nIMPLn' : '(NOT {0}) IMPLIES (NOT {1})',
                'pANDpIMPLp' : '{0} AND {1} IMPLIES {2}', 'pANDpIMPLn' : '{0} AND {1} IMPLIES (NOT {2})', 
                'pANDnIMPLp' : '{0} AND (NOT {1}) IMPLIES {2}', 'pANDnIMPLn' : '{0} AND (NOT {1}) IMPLIES (NOT {2})', 
                'nANDpIMPLp' : '(NOT {0}) AND {1} IMPLIES {2}', 'nANDpIMPLn' : '(NOT {0}) AND {1} IMPLIES (NOT {2})', 
                'nANDnIMPLp' : '(NOT {0}) AND (NOT {1}) IMPLIES {2}', 'nANDnIMPLn' : '(NOT {0}) AND (NOT {1}) IMPLIES (NOT {2})',
                'pIMPLpORp' : '{0} IMPLIES {1} OR {2}', 'pIMPLpORn' : '{0} IMPLIES {1} OR (NOT {2})', 
                'pIMPLnORp' : '{0} IMPLIES (NOT {1}) OR {2}', 'pIMPLnORn' : '{0} IMPLIES (NOT {1}) OR (NOT {2})', 
                'nIMPLpORp' : '(NOT {0}) IMPLIES {1} OR {2}', 'nIMPLpORn' : '(NOT {0}) IMPLIES {1} OR (NOT {2})', 
                'nIMPLnORp' : '(NOT {0}) IMPLIES (NOT {1}) OR {2}', 'nIMPLnORn' : '(NOT {0}) IMPLIES (NOT {1}) OR (NOT {2})',
                'pDIFFp' : '{0} DIFFERENT FROM {1}'}              

# define names of output files
summary_output_file_template = "output/{0}_{1}-rules_{2}.csv"
rules_output_prefix_template = "output/rules_{2}/{0}-{1}"
rules_output_template = "{0}-{1}-{2}.csv"

def evaluate_rules(rules_dict, data_set, config, algorithm, quiet):
    """Evaluates the validity of the rules given in the rules_dict and outputs them (both to console and files)."""

    summary_output_file = summary_output_file_template.format(data_set, config, algorithm)
    rules_output_prefix = rules_output_prefix_template.format(data_set, config, algorithm)
    
    output_strings = ['rule_type,desired_threshold,training_threshold,num_rules,min_test_accuracy,avg_test_accuracy\n']

    if not quiet:
        print("\nRule evaluation")
    
    for rule_type, rule_instances in rules_dict.items():
        for confidence_threshold in [0.7, 0.8, 0.9, 0.95, 0.99]:
            # filter list of rules according to confidence_threshold
            filtered_list = list(filter(lambda x: x[0][1] >= confidence_threshold and x[0][0] >= confidence_threshold, rule_instances))
            if len(filtered_list) == 0:
                if not quiet:
                    print("Could not find rules of type {0} when trying to achieve {1} on validation set.".format(rule_type, confidence_threshold))
                continue
            
            # compute threshold on training data as well as performance on test data
            training_threshold = min(map(lambda x: x[0][0], filtered_list))
            average_test_accuracy = sum(map(lambda x: x[0][2], filtered_list)) / len(filtered_list)
            minimal_test_accuracy = min(map(lambda x: x[0][2], filtered_list))
    
            # print right away        
            if not quiet:
                print("Threshold for rules of type {0} when trying to achieve {1} on validation set: {2} " \
                    "(leaving {3} rules, accuracy on test set: avg {4}, min {5})".format(rule_type, confidence_threshold, 
                      training_threshold, len(filtered_list), average_test_accuracy, minimal_test_accuracy))
            
            # store for output into file
            output_strings.append(",".join([rule_type, str(confidence_threshold), str(training_threshold), str(len(filtered_list)), 
                                            str(minimal_test_accuracy), str(average_test_accuracy)]) + '\n')
            
            # write resulting rules into file
            rules_file_name =  rules_output_template.format(rules_output_prefix, rule_type, confidence_threshold)
            with open(rules_file_name, 'w') as f:
                f.write("rule,training,validation,test\n")
                for rule in filtered_list:
                    if len(rule) == 3:
                        rule_string = rule_strings[rule_type].format(rule[1], rule[2])
                    elif len(rule) == 4:
                        rule_string = rule_strings[rule_type].format(rule[1], rule[2], rule[3])
                    else:
                        raise(Exception("invalid length of rule information"))
                    line = "{0},{1},{2},{3}\n".format(rule_string, rule[0][0], rule[0][1], rule[0][2])
                    f.write(line)
        
        if not quiet:
            print("")
    
    with open(summary_output_file, 'w') as f:
        for line in output_strings:
            f.write(line)    

def clause_results_to_rule_results(clause_results, rule_results):
    """Automatically map clauses of two and three literals into logical rules and modify rule_results accordingly."""

    true_false_map = {True:'p', False:'n'}

    for rule_type in rule_types:
        rule_results[rule_type] = []
    
    for [validities, config] in clause_results:
        concepts = config[0]
        literal_values = config[1]
        
        if len(concepts) == 2:
            rule_type = '{0}IMPL{1}'.format(true_false_map[not literal_values[0]], true_false_map[literal_values[1]])
            rule_results[rule_type].append([validities, concepts[0], concepts[1]])
            if rule_type == 'pIMPLn':
                rule_results['pDIFFp'].append([validities, concepts[0], concepts[1]])

            rule_type = '{0}IMPL{1}'.format(true_false_map[not literal_values[1]], true_false_map[literal_values[0]])
            rule_results[rule_type].append([validities, concepts[1], concepts[0]])
            if rule_type == 'pIMPLn':
                rule_results['pDIFFp'].append([validities, concepts[1], concepts[0]])
                
        elif len(concepts) == 3:
            #  'pANDpIMPLp', 'pANDpIMPLn', 'pANDnIMPLp', 'pANDnIMPLn', 'nANDpIMPLp', 'nANDpIMPLn', 'nANDnIMPLp', 'nANDnIMPLn',
            #  'pIMPLpORp', 'pIMPLpORn', 'pIMPLnORp', 'pIMPLnORn', 'nIMPLpORp', 'nIMPLpORn', 'nIMPLnORp', 'nIMPLnORn',
            permutations = itertools.permutations([0,1,2])
            for [x,y,z] in permutations:
                rule_type_and = '{0}AND{1}IMPL{2}'.format(true_false_map[not literal_values[x]], true_false_map[not literal_values[y]], 
                                                            true_false_map[literal_values[z]])
                rule_results[rule_type_and].append([validities, concepts[x], concepts[y], concepts[z]])                                        
            
                
                rule_type_or = '{0}IMPL{1}OR{2}'.format(true_false_map[not literal_values[x]], true_false_map[literal_values[y]], 
                                                            true_false_map[literal_values[z]])
                rule_results[rule_type_or].append([validities, concepts[x], concepts[y], concepts[z]])    
        else:
            raise(Exception("invalid number of literals in rule clause!"))