# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 2017

@author: lbechberger
"""

import tensorflow as tf
import logictensornetworks as ltn

import sys, re, random

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

# read config file and set up LTN parameters
config = util.parse_config_file(config_file_name, config_name)

def set_ltn_variable(config, name, default):
    if name in config:
        return config[name]
    else:
        return default

ltn.default_layers                  = set_ltn_variable(config, "ltn_layers", ltn.default_layers)
ltn.default_smooth_factor           = set_ltn_variable(config, "ltn_smooth_factor", ltn.default_smooth_factor)
ltn.default_tnorm                   = set_ltn_variable(config, "ltn_tnorm", ltn.default_tnorm)
ltn.default_aggregator              = set_ltn_variable(config, "ltn_aggregator", ltn.default_aggregator)
ltn.default_optimizer               = set_ltn_variable(config, "ltn_optimizer", ltn.default_optimizer)
ltn.default_clauses_aggregator      = set_ltn_variable(config, "ltn_clauses_aggregator", ltn.default_clauses_aggregator)
ltn.default_positive_fact_penality  = set_ltn_variable(config, "ltn_positive_fact_penalty", ltn.default_positive_fact_penality)
ltn.default_norm_of_u               = set_ltn_variable(config, "ltn_norm_of_u", ltn.default_norm_of_u)

# create conceptual space
conceptual_space = ltn.Domain(config["num_dimensions"], label="ConceptualSpace")

# create concepts as predicates
concepts = {}
for label in config["concepts"]:       
    concept = ltn.Predicate(label, conceptual_space)
    concepts[label] = concept


# parse rules file
rules = []
num_rules = 0
implication_rule = re.compile("(\w+) IMPLIES (\w+)")
different_concepts_rule = re.compile("(\w+) DIFFERENT (\w+)")
with open(config["rules_file"], 'r') as f:
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

# additional rules: labeled data points need to be classified correctly
i = 1
for labels, vec in config["training_vectors"]:
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
data = map(lambda x: x[1], config["training_vectors"])
feed_dict = { conceptual_space.tensor : data }

# print out some diagnostic information
print("program arguments: {0} {1}".format(config_file_name, config_name))
print("number of concepts: {0}".format(len(concepts.keys())))
print("number of rules: {0}".format(num_rules))
print("number of training points: {0}".format(len(config["training_vectors"])))
print("number of test points: {0}".format(len(config["test_vectors"])))

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
for i in range(config["max_iter"]):
  KB.train(sess, feed_dict = feed_dict)
  sat_level = sess.run(KB.tensor, feed_dict = feed_dict)
  print(i + 1, " ------> ", sat_level)
  if sat_level > .99:
      break

#KB.save(sess)  # save the result if needed

# evaluate the results: classify each of the test data points
test_data = map(lambda x: x[1], config["test_vectors"])
concept_memberships = {}
train_memberships = {}
for label, concept in concepts.iteritems():
    concept_memberships[label] = np.squeeze(sess.run(concept.tensor(), {conceptual_space.tensor:test_data}))
    train_memberships[label] = np.squeeze(sess.run(concept.tensor(), feed_dict))
    max_membership = max(concept_memberships[label])
    min_membership = min(concept_memberships[label])
    print "{0}: max {1} min {2} - diff {3}".format(label, max_membership, min_membership, max_membership - min_membership)

# compute evaluation measures 
util.evaluate(train_memberships, config["training_vectors"], concept_memberships, config["test_vectors"])

# TODO: auto-generate and check for rules (A IMPLIES B, A DIFFERENT B, etc.)

# visualize the results for 2D and 3D data
if b_plot and config["num_dimensions"] == 2:
    from pylab import cm
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    
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


elif b_plot and config["num_dimensions"] == 3:
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