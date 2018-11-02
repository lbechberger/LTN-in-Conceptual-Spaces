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
import itertools

import util

# fix random seed to ensure reproducibility with respect to usage of negative labels
random.seed(42)

# parse command line arguments
parser = argparse.ArgumentParser(description='LTN in CS')
parser.add_argument('-t', '--type', default = None,
                    help = 'the type of LTN membership function to use')
parser.add_argument('-p', '--plot', action="store_true",
                    help = 'plot the resulting concepts if space is 2D or 3D')
parser.add_argument('-q', '--quiet', action="store_true", help = 'disables info output')                    
parser.add_argument('-e', '--early', action="store_true", help = 'Stop early (if 99% sat)')  
parser.add_argument('-r', '--rules', action="store_true", help = 'Extract rules from the LTN')                  
parser.add_argument('config_file', help = 'the config file to use')
parser.add_argument('config_name', help = 'the name of the configuration')
args = parser.parse_args()

# parse the config file
config = util.parse_config_file(args.config_file, args.config_name)

def set_ltn_variable(config, name, default):
    if name in config:
        return config[name]
    else:
        return default

ltn.default_layers                  = set_ltn_variable(config, "layers", ltn.default_layers)
ltn.default_smooth_factor           = set_ltn_variable(config, "smooth_factor", ltn.default_smooth_factor)
ltn.default_tnorm                   = set_ltn_variable(config, "tnorm", ltn.default_tnorm)
ltn.default_aggregator              = set_ltn_variable(config, "aggregator", ltn.default_aggregator)
ltn.default_optimizer               = set_ltn_variable(config, "optimizer", ltn.default_optimizer)
ltn.default_clauses_aggregator      = set_ltn_variable(config, "clauses_aggregator", ltn.default_clauses_aggregator)
ltn.default_positive_fact_penality  = set_ltn_variable(config, "positive_fact_penalty", ltn.default_positive_fact_penality)
ltn.default_norm_of_u               = set_ltn_variable(config, "norm_of_u", ltn.default_norm_of_u)
ltn.default_epsilon                 = set_ltn_variable(config, "epsilon", ltn.default_epsilon)
if args.type != None:
    ltn.default_type = args.type
else:
    ltn.default_type = set_ltn_variable(config, "type", ltn.default_type)


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
literals = {}
rules = []
feed_dict = {}
for label in config["concepts"]:
    
    concepts[label] = ltn.Predicate(label, conceptual_space, data_points = pos_examples[label])
    pos_literal = ltn.Literal(True, concepts[label], conceptual_space)
    neg_literal = ltn.Literal(False, concepts[label], conceptual_space)
    literals[label] = {True: pos_literal, False: neg_literal}
    
    # it can happen that we don't have any positive examples; then: don't try to add a rule
    if len(pos_examples[label]) > 0:
        pos_domain = ltn.Domain(conceptual_space.columns, label = label + "_pos_ex")
        rules.append(ltn.Clause([ltn.Literal(True, concepts[label], pos_domain)], label="{0}Const".format(label), weight = 1.0*len(pos_examples[label])))
        feed_dict[pos_domain.tensor] = pos_examples[label]
        
    if len(neg_examples[label]) > 0:    
        neg_domain = ltn.Domain(conceptual_space.columns, label = label + "_neg_ex")
        rules.append(ltn.Clause([ltn.Literal(False, concepts[label], neg_domain)], label="{0}ConstNot".format(label), weight = 1.0*len(neg_examples[label])))
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
            rules.append(ltn.Clause([literals[left][False], literals[right][True]], label = "{0}I{1}".format(left, right)))
            num_rules += 1
        else:
            matches = different_concepts_rule.findall(line)
            if len(matches) > 0:
                left = matches[0][0]
                right = matches[0][1]
                # 'left DIFFERENT right' <--> 'NOT (left AND right)' <--> (NOT left) OR (NOT right)'
                rules.append(ltn.Clause([literals[left][False], literals[right][True]], label = "{0}DC{1}".format(left, right)))
                num_rules += 1

# only extract rules if explicitly asked for
if args.rules:
        
    # list of clauses representing candidate rules
    rule_clause_tensors = []
    
    # rules involving two concepts
    for [first_concept, second_concept] in itertools.combinations(config["concepts"], 2):
        
        literal_values = itertools.product([True, False], repeat=2)
        
        for [first_val, second_val] in literal_values:
            tensor = ltn.Clause([literals[first_concept][first_val], literals[second_concept][second_val]], 
                                              label = "{0}{1}O{2}{3}".format(first_val, first_concept, 
                                                        second_val, second_concept)).tensor
            rule_clause_tensors.append([tensor, [first_concept, second_concept], [first_val, second_val]])
    
    # rules involving three concepts
    for [first_concept, second_concept, third_concept] in itertools.combinations(config["concepts"], 3):
        
        literal_values = itertools.product([True, False], repeat=3)
        
        for [first_val, second_val, third_val] in literal_values:
            tensor = ltn.Clause([literals[first_concept][first_val], literals[second_concept][second_val], 
                                               literals[third_concept][third_val]], 
                                              label = "{0}{1}O{2}{3}O{4}{5}".format(first_val, first_concept, 
                                                        second_val, second_concept, third_val, third_concept)).tensor
            rule_clause_tensors.append([tensor, [first_concept, second_concept, third_concept], [first_val, second_val, third_val]])


# grab training, validation, and test data
training_data = list(map(lambda x: x[1], config["training_vectors"]))
validation_data = list(map(lambda x: x[1], config["validation_vectors"]))
test_data = list(map(lambda x: x[1], config["test_vectors"]))

# train over training data
feed_dict[conceptual_space.tensor] = training_data

# print out some diagnostic information
if not args.quiet:
    print("program arguments: {0}".format(args))
    print("number of concepts: {0}".format(len(concepts.keys())))
    print("number of rules: {0}".format(num_rules))
    print("number of training points: {0}".format(len(config["training_vectors"])))
    print("number of validation points: {0}".format(len(config["validation_vectors"])))
    print("number of test points: {0}".format(len(config["test_vectors"])))
    if args.rules:
        print("Generated {0} candidate rules.".format(len(rule_clause_tensors)))

# knowledge base = set of all clauses (all of them should be optimized)
KB = ltn.KnowledgeBase("ConceptualSpaceKB", rules, "")
# initialize LTN and TensorFlow
init = tf.global_variables_initializer()
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

sess = tf.Session(config=tf_config)
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

if not args.quiet:
    print(0, " ------> ", sat_level)

# train for at most max_iter iterations (stop if we hit 99% satisfiability earlier and flag set)
max_iter = max(config['epochs'])
if not args.quiet:
    print("Epochs to check: {0}".format(config['epochs']))

for i in range(max_iter):
    buf = KB.train(sess, feed_dict = feed_dict)
    sat_level = sess.run(KB.tensor, feed_dict = feed_dict)
    if not args.quiet:
        print(i + 1, " ------> ", sat_level)
    if sat_level > .99 and args.early:
        break
    
    if i + 1 in config['epochs']:
        if not args.quiet:
            print("\nResults After Epoch {0}".format(i+1))
            print("===========================")
            
        # evaluate the results: classify each of the test data points
        training_memberships = {}
        validation_memberships = {}
        test_memberships = {}        
        
        all_memberships_identical = True
        standard_membership = 9999
        all_spreads_small = True
        
        all_concept_tensors = []
        for label in config["concepts"]:
            all_concept_tensors.append(concepts[label].tensor())
        
        train_results = np.squeeze(sess.run(all_concept_tensors, {conceptual_space.tensor:training_data}))
        valid_results = np.squeeze(sess.run(all_concept_tensors, {conceptual_space.tensor:validation_data}))
        test_results = np.squeeze(sess.run(all_concept_tensors, {conceptual_space.tensor:test_data}))
        
        for idx, label in enumerate(config["concepts"]):
            training_memberships[label] = train_results[idx]
            validation_memberships[label] = valid_results[idx]
            test_memberships[label] = test_results[idx]        
            
            max_membership = max(validation_memberships[label])
            min_membership = min(validation_memberships[label])

            spread = max_membership - min_membership
            if spread > 1e-8:
                all_spreads_small = False
            
            if standard_membership == 9999:
                standard_membership = max_membership
            if max_membership != standard_membership or min_membership != standard_membership:
                    all_memberships_identical = False
            if not args.quiet: 
                print("{0}: max {1} min {2} - diff {3}".format(label, max_membership, min_membership, spread))
        
        # if all memberships are identical (e.g., all 0.5 or all NaN), then there's no point in evaluating the system
        if all_memberships_identical or all_spreads_small:
            print("LTN has collapsed! not evaluating.")
        else:
            # compute evaluation measures 
            eval_results = {}
            eval_results['contents'] = ['training', 'validation', 'test']
            eval_results['training'] = util.evaluate(training_memberships, config["training_vectors"], config["concepts"])
            eval_results['validation'] = util.evaluate(validation_memberships, config["validation_vectors"], config["concepts"])
            eval_results['test'] = util.evaluate(test_memberships, config["test_vectors"], config["concepts"])
            if not args.quiet:
                util.print_evaluation(eval_results)
            util.write_evaluation(eval_results, "output/{0}-LTN.csv".format(args.config_file.split('.')[0]), "{0}-ep{1}".format(args.config_name, i + 1))
    
            # check for rules if asked to do so
            if args.rules:
                rule_tensors = list(map(lambda x: x[0], rule_clause_tensors))
                training_results = np.squeeze(sess.run(rule_tensors, {conceptual_space.tensor:training_data}))
                validation_results = np.squeeze(sess.run(rule_tensors, {conceptual_space.tensor:validation_data}))
                test_results = np.squeeze(sess.run(rule_tensors, {conceptual_space.tensor:test_data}))
                
                all_data_results = list(zip(training_results, validation_results, test_results))
                
                clause_results = list(zip(all_data_results, list(map(lambda x: x[1:], rule_clause_tensors))))
                
                rule_results = {}
                util.clause_results_to_rule_results(clause_results, rule_results)
                util.evaluate_rules(rule_results, args.config_file.split('.')[0], args.config_name, 'LTN', args.quiet)
                
            # check for rules
#            rule_results = {}
#            for rule_type, vectors in generated_rules.items():
#                
#                tensors = list(map(lambda x: x[0], vectors))
#                training_results = np.squeeze(sess.run(tensors, {conceptual_space.tensor:training_data}))
#                validation_results = np.squeeze(sess.run(tensors, {conceptual_space.tensor:validation_data}))
#                test_results = np.squeeze(sess.run(tensors, {conceptual_space.tensor:test_data}))
#                
#                local_results = []
#                for idx, entry in enumerate(vectors):
#                    local_results.append([[training_results[idx], validation_results[idx], test_results[idx]]] + entry[1:])
#                
#                rule_results[rule_type] = local_results
#            
#            util.evaluate_rules(rule_results, args.config_file.split('.')[0], args.config_name, 'LTN', args.quiet)

# visualize the results for 2D and 3D data if flag is set
if args.plot and config["num_dimensions"] == 2:
    import matplotlib
    matplotlib.use('TkAgg')
    from pylab import cm
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
    import matplotlib
    matplotlib.use('TkAgg')
    from pylab import cm
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
    for label, memberships in validation_memberships.items():
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
                
                x = np.array(x)
                y = np.array(y)
                z = np.array(z)                
                
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