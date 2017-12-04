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

from math import sqrt

import util

# fix random seed to ensure reproducibility
random.seed(42)

# parse command line arguments
if len(sys.argv) < 3:
    raise Exception("Need two arguments: 'python run_ltn.py config.cfg config_name'")
config_file_name = sys.argv[1]
config_name = sys.argv[2]

if len(sys.argv) > 3 and sys.argv[3] == '-p':
    b_plot = True
else:
    b_plot = False
# read configuartion from the given config file
config = ConfigParser.RawConfigParser()
config.read(config_file_name)

# general setup
features_file = config.get(config_name, "features_file")
concepts_file = config.get(config_name, "concepts_file")
rules_file = config.get(config_name, "rules_file")
n_dims = config.getint(config_name, "num_dimensions")
training_percentage = config.getfloat(config_name, "training_percentage")
max_iter = config.getint(config_name, "max_iter")

# LTN setup
def read_ltn_variable(name, default):
    if config.has_option(config_name, name):
        return config.get(config_name, name)
    elif config.has_option("ltn-default", name):
        return config.get("ltn-default", name)  
    else:
        return default
        
ltn.default_layers                  = int(read_ltn_variable("ltn_layers", ltn.default_layers))
ltn.default_smooth_factor           = float(read_ltn_variable("ltn_smooth_factor", ltn.default_smooth_factor))     
ltn.default_tnorm                   = read_ltn_variable("ltn_tnorm", ltn.default_tnorm)
ltn.default_aggregator              = read_ltn_variable("ltn_aggregator", ltn.default_aggregator)        
ltn.default_optimizer               = read_ltn_variable("ltn_optimizer", ltn.default_optimizer)
ltn.default_clauses_aggregator      = read_ltn_variable("ltn_clauses_aggregator", ltn.default_clauses_aggregator)
ltn.default_positive_fact_penality  = float(read_ltn_variable("ltn_positive_fact_penalty", ltn.default_positive_fact_penality))
ltn.default_norm_of_u               = float(read_ltn_variable("ltn_norm_of_u", ltn.default_norm_of_u))

# create conceptual space
conceptual_space = ltn.Domain(n_dims, label="ConceptualSpace")

# parse features file
feature_vectors = []
with open(features_file, 'r') as f:
    for line in f:
        chunks = line.replace('\n','').replace('\r', '').split(",")
        vec = map(float, chunks[:n_dims])
        labels = [label for label in chunks[n_dims:] if label != '']
        feature_vectors.append((labels, vec))
# shuffle them --> beginning of shuffle list will be treated as labeled, end as unlabeled
random.shuffle(feature_vectors)

# parse concepts file
concepts = {}
with open(concepts_file, 'r') as f:
    for line in f:       
        name = line.replace('\n','').replace('\r', '')
        concept = ltn.Predicate(name, conceptual_space)
        concepts[name] = concept

# parse rules file
rules = []
num_rules = 0
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
            num_rules += 1
        else:
            matches = different_concepts_rule.findall(line)
            if len(matches) > 0:
                left = matches[0][0]
                right = matches[0][1]
                # 'left DIFFERENT right' <--> 'NOT (left AND right)' <--> (NOT left) OR (NOT right)'
                rules.append(ltn.Clause([ltn.Literal(False, concepts[left], conceptual_space), 
                                     ltn.Literal(False, concepts[right], conceptual_space)], label = "{0}DC{1}".format(left, right)))
                num_rules += 1

# sample training_percentage of the data points as labeled ones
cutoff = int(len(feature_vectors) * training_percentage)
training_vectors = feature_vectors[:cutoff]
test_vectors = feature_vectors[cutoff:]

# add rules: labeled data points need to be classified correctly
i = 1
for labels, vec in training_vectors:
    const = ltn.Constant("_".join(map(str, vec)), vec, conceptual_space)
    for label in labels:
        # classify under correct labels
        rules.append(ltn.Clause([ltn.Literal(True, concepts[label], const)], label="{0}Const".format(label)))
    
    # don't classify under incorrect label (pick a random one)
    possible_negative_labels = list(set(concepts.keys()) - set(labels))
    negative_label = random.choice(possible_negative_labels)
    rules.append(ltn.Clause([ltn.Literal(False, concepts[negative_label], const)], label="{0}ConstNot".format(negative_label)))
    
    if i % 1000 == 0:
        print i
    i += 1
    

# all data points in the conceptual space over which we would like to optimize:
data = map(lambda x: x[1], training_vectors)
feed_dict = { conceptual_space.tensor : data }

# print out some diagnostic information
print("program arguments: {0} {1}".format(config_file_name, config_name))
print("number of concepts: {0}".format(len(concepts.keys())))
print("number of rules: {0}".format(num_rules))
print("number of data points: {0}".format(len(feature_vectors)))

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

# train for at most max_iter iterations (stop if we hit 99% satisfiability earlier)
for i in range(max_iter):
  KB.train(sess, feed_dict = feed_dict)
  sat_level = sess.run(KB.tensor, feed_dict = feed_dict)
  print(i + 1, " ------> ", sat_level)
  if sat_level > .99:
      break

#KB.save(sess)  # save the result if needed

# evaluate the results: classify each of the test data points
test_data = map(lambda x: x[1], test_vectors)
concept_memberships = {}
train_memberships = {}
for label, concept in concepts.iteritems():
    concept_memberships[label] = np.squeeze(sess.run(concept.tensor(), {conceptual_space.tensor:test_data}))
    train_memberships[label] = np.squeeze(sess.run(concept.tensor(), feed_dict))
    max_membership = max(concept_memberships[label])
    min_membership = min(concept_memberships[label])
    print "{0}: max {1} min {2} - diff {3}".format(label, max_membership, min_membership, max_membership - min_membership)

# compute evaluation measures 
print("One error on training data: {0}".format(util.one_error(train_memberships, training_vectors)))
print("Coverage on training data: {0}".format(util.coverage(train_memberships, training_vectors)))
print("One error on test data: {0}".format(util.one_error(concept_memberships, test_vectors)))
print("Coverage on test data: {0}".format(util.coverage(concept_memberships, test_vectors)))

# TODO: auto-generate and check for rules (A IMPLIES B, A DIFFERENT B, etc.)

# visualize the results for 2D and 3D data
if b_plot and n_dims == 2:
    from pylab import cm
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(16,10))
    xs = map(lambda x: x[0], test_data)
    ys = map(lambda x: x[1], test_data)

    # figure out how many subplots to create (rows and columns)
    num_concepts = len(concepts)
    root = int(sqrt(num_concepts))
    if root * root >= num_concepts:
        columns = root
        rows = root
    elif root * (root + 1) >= num_concepts:
        columns = root + 1
        rows = root
    else:
        columns = root + 1
        rows = root
    
    # for each concept, create a colored scatter plot of all unlabeled data points
    counter = 1
    for label, memberships in concept_memberships.iteritems():
        colors = cm.jet(memberships)
        colmap = cm.ScalarMappable(cmap=cm.jet)
        colmap.set_array(memberships)
        ax = fig.add_subplot(rows, columns, counter)
        ax.set_title(label)
        yg = ax.scatter(xs, ys, c=colors, marker='o')
        cb = fig.colorbar(colmap)
        counter += 1
   
    plt.show()


elif b_plot and n_dims == 3:
    from pylab import cm
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(16,10))
    xs = map(lambda x: x[0], test_data)
    ys = map(lambda x: x[1], test_data)
    zs = map(lambda x: x[2], test_data)

    # figure out how many subplots to create (rows and columns)
    num_concepts = len(concepts)
    root = int(sqrt(num_concepts))
    if root * root >= num_concepts:
        columns = root
        rows = root
    elif root * (root + 1) >= num_concepts:
        columns = root + 1
        rows = root
    else:
        columns = root + 1
        rows = root
    
    # for each concept, create a colored scatter plot of all unlabeled data points
    counter = 1
    for label, memberships in concept_memberships.iteritems():
        colors = cm.jet(memberships)
        colmap = cm.ScalarMappable(cmap=cm.jet)
        colmap.set_array(memberships)
        ax = fig.add_subplot(rows, columns, counter, projection='3d')
        ax.set_title(label)
        yg = ax.scatter(xs, ys, zs, c=colors, marker='o')
        cb = fig.colorbar(colmap)
        counter += 1
        
    plt.show()

# close the TensorFlow session and go home
sess.close()