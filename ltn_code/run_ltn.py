# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 2017

@author: lbechberger
"""

import tensorflow as tf
import logictensornetworks as ltn
import numpy as np

import re, random, argparse
from math import sqrt, isnan

import util

# fix random seed to ensure reproducibility
random.seed(42)

# parse command line arguments
parser = argparse.ArgumentParser(description='LTN in CS')
parser.add_argument('-t', '--type', default = 'original',
                    help = 'the type of LTN membership function to use')
parser.add_argument('-p', '--plot', action="store_true",
                    help = 'plot the resulting concepts if space is 2D or 3D')
parser.add_argument('-q', '--quiet', action="store_true",
                    help = 'disables info output')                    
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
prototypes = {}
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
training_data = list(map(lambda x: x[1], config["training_vectors"]))
feed_dict[conceptual_space.tensor] = training_data

# print out some diagnostic information
if not args.quiet:
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
if not args.quiet:
    print("initialization", sat_level)

if ltn.default_type != "cuboid":
    # if first initialization was horrible: re-try until we get something reasonable
    while sat_level < 1e-10:
        sess.run(init)
        sat_level = sess.run(KB.tensor, feed_dict = feed_dict)
        if not args.quiet:
            print("initialization",sat_level)
print(0, " ------> ", sat_level)


# train for at most max_iter iterations (stop if we hit 99% satisfiability earlier)
for i in range(config["max_iter"]):
    buf = KB.train(sess, feed_dict = feed_dict)
    sat_level = sess.run(KB.tensor, feed_dict = feed_dict)
    if not args.quiet:
        print(i + 1, " ------> ", sat_level)
    if sat_level > .99:
        break

#KB.save(sess)  # save the result if needed

# evaluate the results: classify each of the test data points
validation_data = list(map(lambda x: x[1], config["validation_vectors"]))
validation_memberships = {}
training_memberships = {}
for label, concept in concepts.items():
    validation_memberships[label] = np.squeeze(sess.run(concept.tensor(), {conceptual_space.tensor:validation_data}))
    training_memberships[label] = np.squeeze(sess.run(concept.tensor(), {conceptual_space.tensor:training_data}))
    if not args.quiet:
        max_membership = max(validation_memberships[label])
        min_membership = min(validation_memberships[label])
        print("{0}: max {1} min {2} - diff {3}".format(label, max_membership, min_membership, max_membership - min_membership))

# compute evaluation measures 
eval_results = {}
eval_results['training'] = util.evaluate(training_memberships, config["training_vectors"], config["concepts"])
eval_results['validation'] = util.evaluate(validation_memberships, config["validation_vectors"], config["concepts"])
if not args.quiet:
    util.print_evaluation(eval_results)
util.write_evaluation(eval_results, "output/{0}-LTN.csv".format(args.config_file.split('.')[0]), args.config_name)

#if ltn.default_type == 'cuboid':
#    for label, concept in concepts.items():
#        prot, first, second, p1, p2, p_min, p_max, c, weights = sess.run([concept.prototype, concept.first_vector, concept.second_vector, concept.point_1, concept.point_2, concept.p_min, concept.p_max, concept.c, concept.weights], feed_dict = {conceptual_space.tensor:training_data})
#        print label
#        print "prototype: {0}".format(prot)
#        print "first_vector: {0}".format(first)
#        print "second_vector: {0}".format(second)
#        print "point_1: {0}".format(p1)
#        print "point_2: {0}".format(p2)
#        print "p_min: {0}".format(p_min)
#        print "p_max: {0}".format(p_max)
#        print "c: {0}".format(c)
#        print "weights: {0}".format(weights)
#        print " "

# TODO: auto-generate and check for rules (A IMPLIES B, A DIFFERENT B, etc.)

# visualize the results for 2D and 3D data
if args.plot and config["num_dimensions"] == 2:
    from pylab import cm
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(16,10))
    xs = list(map(lambda x: x[0], validation_data))
    ys = list(map(lambda x: x[1], validation_data))

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
    for label, memberships in validation_memberships.items():
        colors = cm.jet(memberships)
        colmap = cm.ScalarMappable(cmap=cm.jet)
        colmap.set_array(memberships)
        ax = fig.add_subplot(rows, columns, counter)
        
        if ltn.default_type == "cuboid":
            # also plot the actual box
            import matplotlib.patches as patches
            import shapely.geometry
            from matplotlib.path import Path
            
            def _path_for_core(cuboids, d1, d2):
                """Creates the 2d path for a complete core."""
            
                polygon = None    
                for cuboid in cuboids:
                    p_min = cuboid[0]
                    p_max = cuboid[1]
                    cub = shapely.geometry.box(p_min[d1], p_min[d2], p_max[d1], p_max[d2])
                    if polygon == None:
                        polygon = cub
                    else:
                        polygon = polygon.union(cub)
                
                verts = list(polygon.exterior.coords)
                codes = [Path.LINETO] * len(verts)
                codes[0] = Path.MOVETO
                codes[-1] = Path.CLOSEPOLY
                
                path = Path(verts, codes)
                return path
                
            p_min = sess.run(concepts[label].p_min)
            p_max = sess.run(concepts[label].p_max)
            cuboids = zip(p_min, p_max)
            if not isnan(p_min[0][0]):
                core_path = _path_for_core(cuboids, 0, 1)
                core_patch = patches.PathPatch(core_path, facecolor='grey', lw=2, alpha=0.4)
                ax.add_patch(core_patch)
        elif ltn.default_type == "prototype":
            # plot the prototype
            prototypes = sess.run(concepts[label].prototypes)
            for p in prototypes:
                ax.scatter(p[0], p[1], marker='x')
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
    xs = list(map(lambda x: x[0], validation_data))
    ys = list(map(lambda x: x[1], validation_data))
    zs = list(map(lambda x: x[2], validation_data))

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
    for label, memberships in training_memberships.items():#validation_memberships.items():
        colors = cm.jet(memberships)
        colmap = cm.ScalarMappable(cmap=cm.jet)
        colmap.set_array(memberships)
        ax = fig.add_subplot(rows, columns, counter, projection='3d')
        
        if ltn.default_type == "cuboid":
            # also plot the actual box
            def _cuboid_data_3d(p_min, p_max):
                """Returns the 3d information necessary for plotting a 3d cuboid."""
                
                a = 0
                b = 1
                c = 2    
                
                x = [[p_min[a], p_max[a], p_max[a], p_min[a], p_min[a]],  # bottom
                     [p_min[a], p_max[a], p_max[a], p_min[a], p_min[a]],  # top
                     [p_min[a], p_max[a], p_max[a], p_min[a], p_min[a]],  # front
                     [p_min[a], p_max[a], p_max[a], p_min[a], p_min[a]]]  # back
                     
                y = [[p_min[b], p_min[b], p_max[b], p_max[b], p_min[b]],  # bottom
                     [p_min[b], p_min[b], p_max[b], p_max[b], p_min[b]],  # top
                     [p_min[b], p_min[b], p_min[b], p_min[b], p_min[b]],  # front
                     [p_max[b], p_max[b], p_max[b], p_max[b], p_max[b]]]  # back
                     
                z = [[p_min[c], p_min[c], p_min[c], p_min[c], p_min[c]],  # bottom
                     [p_max[c], p_max[c], p_max[c], p_max[c], p_max[c]],  # top
                     [p_min[c], p_min[c], p_max[c], p_max[c], p_min[c]],  # front
                     [p_min[c], p_min[c], p_max[c], p_max[c], p_min[c]]]  # back
                
                return x, y, z
            
            p_min = sess.run(concepts[label].p_min)
            p_max = sess.run(concepts[label].p_max)
            for i in range(ltn.default_layers):
                x,y,z = _cuboid_data_3d(p_min[i], p_max[i])
                ax.plot_surface(x, y, z, color="grey", rstride=1, cstride=1, alpha=0.1)
        elif ltn.default_type == "prototype":
            # plot the prototype
            prototypes = sess.run(concepts[label].prototypes)
            for p in prototypes:
                ax.scatter(p[0], p[1], p[2], marker='x')
        ax.set_title(label)
        yg = ax.scatter(xs, ys, zs, c=colors, marker='o')
        cb = fig.colorbar(colmap)
        counter += 1
        
            
    plt.show()

# close the TensorFlow session and go home
sess.close()