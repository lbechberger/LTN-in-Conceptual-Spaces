# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 2017

@author: lbechberger
"""

import tensorflow as tf
import logictensornetworks as ltn
import numpy as np

import re, random, argparse
from math import sqrt

import util

# fix random seed to ensure reproducibility
random.seed(42)

# parse command line arguments
parser = argparse.ArgumentParser(description='LTN in CS')
parser.add_argument('-t', '--type', default = 'original',
                    help = 'the type of LTN membership function to use')
parser.add_argument('-p', '--plot', action="store_true",
                    help = 'plot the resulting concepts if space is 2D or 3D')
parser.add_argument('config_file', help = 'the config file to use')
parser.add_argument('config_name', help = 'the name of the configuration')
args = parser.parse_args()

# parse the config file and set the LTN membership function type
config = util.parse_config_file(args.config_file, args.config_name)
ltn.default_type = args.type

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
ltn.default_epsilon                 = set_ltn_variable(config, "ltn_epsilon", ltn.default_epsilon)

# create conceptual space
conceptual_space = ltn.Domain(config["num_dimensions"], label="ConceptualSpace")

# prepare classification rules: labeled data points need to be classified correctly
pos_examples = {}
neg_examples = {}
for label in config["concepts"]:
    pos_examples[label] = []
    neg_examples[label] = []

for labels, vec in config["training_vectors"]:
    for label in labels:
        # classify under correct labels
        pos_examples[label].append(vec)
    
    # don't classify under incorrect label (pick a random one)
    possible_negative_labels = list(set(config["concepts"]) - set(labels))
    negative_label = random.choice(possible_negative_labels)
    neg_examples[negative_label].append(vec)


# create concepts as predicates and create classification rules
concepts = {}
rules = []
feed_dict = {}
for label in config["concepts"]:
    
    concepts[label] = ltn.Predicate(label, conceptual_space, data_points = pos_examples[label])

    # it can happen that we don't have any positive examples; then: don't try to add a rule
    if len(pos_examples[label]) > 0:
        pos_domain = ltn.Domain(conceptual_space.columns, label = label + "_pos_ex")
        rules.append(ltn.Clause([ltn.Literal(True, concepts[label], pos_domain)], label="{0}Const".format(label), weight = len(pos_examples[label])))
        feed_dict[pos_domain.tensor] = pos_examples[label]
        
    if len(neg_examples[label]) > 0:    
        neg_domain = ltn.Domain(conceptual_space.columns, label = label + "_neg_ex")
        rules.append(ltn.Clause([ltn.Literal(False, concepts[label], neg_domain)], label="{0}ConstNot".format(label), weight = len(neg_examples[label])))
        feed_dict[neg_domain.tensor] = neg_examples[label]

# parse rules file
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
   

# all data points in the conceptual space over which we would like to optimize:
data = map(lambda x: x[1], config["training_vectors"])
feed_dict[conceptual_space.tensor] = data

# print out some diagnostic information
print("program arguments: {0}".format(args))
print("number of concepts: {0}".format(len(concepts.keys())))
print("number of rules: {0}".format(num_rules))
print("number of training points: {0}".format(len(config["training_vectors"])))
print("number of validation points: {0}".format(len(config["validation_vectors"])))
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
validation_data = map(lambda x: x[1], config["validation_vectors"])
validation_memberships = {}
training_memberships = {}
for label, concept in concepts.iteritems():
    validation_memberships[label] = np.squeeze(sess.run(concept.tensor(), {conceptual_space.tensor:validation_data}))
#    print validation_memberships[label]
#    print validation_data
    training_memberships[label] = np.squeeze(sess.run(concept.tensor(), feed_dict))
    max_membership = max(validation_memberships[label])
    min_membership = min(validation_memberships[label])
    print "{0}: max {1} min {2} - diff {3}".format(label, max_membership, min_membership, max_membership - min_membership)

# compute evaluation measures 
util.evaluate(training_memberships, config["training_vectors"], validation_memberships, config["validation_vectors"])

# TODO: auto-generate and check for rules (A IMPLIES B, A DIFFERENT B, etc.)

# visualize the results for 2D and 3D data
if args.plot and config["num_dimensions"] == 2:
    from pylab import cm
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(16,10))
    xs = map(lambda x: x[0], validation_data)
    ys = map(lambda x: x[1], validation_data)

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
    for label, memberships in validation_memberships.iteritems():
        colors = cm.jet(memberships)
        colmap = cm.ScalarMappable(cmap=cm.jet)
        colmap.set_array(memberships)
        ax = fig.add_subplot(rows, columns, counter)
        ax.set_title(label)
        yg = ax.scatter(xs, ys, c=colors, marker='o')
        cb = fig.colorbar(colmap)
        counter += 1
   
    plt.show()


elif args.plot and config["num_dimensions"] == 3:
    from pylab import cm
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(16,10))
    xs = map(lambda x: x[0], validation_data)
    ys = map(lambda x: x[1], validation_data)
    zs = map(lambda x: x[2], validation_data)

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
    for label, memberships in validation_memberships.iteritems():
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