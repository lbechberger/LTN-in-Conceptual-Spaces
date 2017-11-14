# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 2017

@author: lbechberger
"""

import tensorflow as tf
import logictensornetworks as ltn
import sys, re, random
import ConfigParser
import numpy as np

# fix random seed to ensure reproducibility
random.seed(42)

#############
# LTN setup #
#############

# number of RBF kernels per predicate; default: 5
ltn.default_layers = 1                 
# TODO: comment; default: 0.0000001
ltn.default_smooth_factor = 1e-10       
# options: 'product', 'yager2', 'luk', 'goedel'; default: 'product'
ltn.default_tnorm = "product"  
# aggregate within a predicate (for complex ones); options: 'product', 'mean', 'gmean', 'hmean', 'min'; default: 'min'         
ltn.default_aggregator = "min"        
# optimizing algorithm to use; options: 'ftrl', 'gd', 'ada', 'rmsprop'; default: 'gd' 
ltn.default_optimizer = "rmsprop"    
# aggregate over clauses, i.e., rules; options: 'min', 'mean', 'hmean', 'wmean'; default: 'min'   
ltn.default_clauses_aggregator = "min"  
# TODO: comment; default: 1e-6
ltn.default_positive_fact_penality = 0  


if sys.argv < 3:
    raise Exception("Need two arguments: 'python run_ltn.py config.cfg config_name'")

config_file_name = sys.argv[1]
config_name = sys.argv[2]

# read configuartion from the given config file
config = ConfigParser.RawConfigParser()
config.read(config_file_name)
features_file = config.get(config_name, "features_file")
concepts_file = config.get(config_name, "concepts_file")
rules_file = config.get(config_name, "rules_file")
n_dims = config.getint(config_name, "num_dimensions")
sample_rate = config.getfloat(config_name, "sample_rate")

# create conceptual space
conceptual_space = ltn.Domain(n_dims, label="ConceptualSpace")

# parse features file
feature_vectors = []
with open(features_file, 'r') as f:
    for line in f:
        chunks = line.split(",")
        label = chunks[0]
        if label == '':
            label = None
        vec = map(float, chunks[1:])
        if len(vec) != n_dims:
            raise Exception("Incorrect dimensionality: Expected {0}, found {1} in: {2}".format(n_dims, len(vec), line))
        feature_vectors.append((label, vec))
# shuffle them --> beginning of shuffle list will be treated as labeled, end as unlabeled
random.shuffle(feature_vectors)

# parse concepts file
concepts = {}
with open(concepts_file, 'r') as f:
    for line in f:       
        name = line.replace('\n','')
        concept = ltn.Predicate(name, conceptual_space)
        concepts[name] = concept

# parse rules file
rules = []
implication_rule = re.compile("(\w+) IMPLIES (\w+)")
different_concepts_rule = re.compile("(\w+) DIFFERENT (\w+)")
with open(rules_file, 'r') as f:
    for line in f:
        matches = implication_rule.findall(line)
        if len(matches) > 0:
            left = matches[0][0]
            right = matches[0][1]
            # 'left IMPLIES right' <--> '(NOT left) OR right'
            rules.append(ltn.Clause([ltn.Literal(False, concepts[left], conceptual_space), 
                                     ltn.Literal(True, concepts[right], conceptual_space)], label = "{0}I{1}".format(left, right)))
        else:
            matches = different_concepts_rule.findall(line)
            if len(matches) > 0:
                left = matches[0][0]
                right = matches[0][1]
                # 'left DIFFERENT right' <--> 'NOT (left AND right)' <--> (NOT left) OR (NOT right)'
                rules.append(ltn.Clause([ltn.Literal(False, concepts[left], conceptual_space), 
                                     ltn.Literal(False, concepts[right], conceptual_space)], label = "{0}DC{1}".format(left, right)))

# sample sample_percent of the data points as labeled ones
cutoff = int(len(feature_vectors) * sample_rate)
labeled_feature_vectors = feature_vectors[:cutoff]
unlabeled_feature_vectors = feature_vectors[cutoff:]

# add rules: labeled data points need to be classified correctly
for label, vec in labeled_feature_vectors:
    const = ltn.Constant(label, vec, conceptual_space)
    rules.append(ltn.Clause([ltn.Literal(True, concepts[label], const)], label="{0}Const".format(label)))

# all data points in the conceptual space over which we would like to optimize
data = map(lambda x: x[1], feature_vectors)
feed_dict = { conceptual_space.tensor : data }

# knowledge base = set of all clauses (all of them should be optimized)
KB = ltn.KnowledgeBase("ConceptualSpaceKB", rules, "")

# initialize LTN and TensorFlow
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
sat_level = sess.run(KB.tensor, feed_dict = feed_dict)
print("initialization", sat_level)

# if first initialization was horrible: re-try until we get something reasonable
while sat_level < 1e-10:
    sess.run(init)
    sat_level = sess.run(KB.tensor, feed_dict = feed_dict)
    print "initialization",sat_level
print(0, " ------> ", sat_level)

# train for at most 1000 iterations (stop if we hit 99% satisfiability earlier)
for i in range(1000):
  KB.train(sess, feed_dict = feed_dict)
  sat_level = sess.run(KB.tensor, feed_dict = feed_dict)
  print(i + 1, " ------> ", sat_level)
  if sat_level > .99:
      break

#KB.save(sess)  # save the result if needed

# evaluate the results: classify each of the unlabeled data points and compute our accuracy
test_data = map(lambda x: x[1], unlabeled_feature_vectors)
concept_memberships = {}
for label, concept in concepts.iteritems():
    concept_memberships[label] = np.squeeze(sess.run(concept.tensor(),{conceptual_space.tensor:test_data}))

idx = 0
num_correct = 0
for (true_label, vector) in unlabeled_feature_vectors:
    predicted_label = None
    predicted_confidence = 0
    for label, memberships in concept_memberships.iteritems():
        conf = memberships[idx]
        if conf > predicted_confidence:
            predicted_confidence = conf
            predicted_label = label
    if predicted_label == true_label:
        num_correct += 1
    idx += 1

accuracy = (1.0 * num_correct) / len(unlabeled_feature_vectors)
print "Overall accuracy: {0}".format(accuracy)

# TODO: visualize the results for 2D and 3D data
if n_dims == 2:
    pass
elif n_dims == 3:
    pass

# close the TensorFlow session and go home
sess.close()