# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 16:49:36 2017

Convert the files provided on https://www.cs.cf.ac.uk/semanticspaces/ into our required format.

Usage like this: python Schockaert_converter.py source_folder n_dims target_folder list_of_classes_to_remove

list_of_classes_to_remove should be one string, with class names separated by commas.

@author: lbechberger
"""

import sys, os

source_folder = sys.argv[1]
n_dims = int(sys.argv[2])
target_folder = sys.argv[3]
to_remove = sys.argv[4]

genre_folder_name = os.path.join(source_folder, "classesGenres")
vector_file_name = os.path.join(source_folder, "films{0}.mds".format(n_dims))
genres_to_ignore = to_remove.split(',')

records = []

with open(vector_file_name, 'r') as f:
    for line in f:
        vector = line.replace('\n', '').split('\t')
        records.append(vector)

genres = []
for file_name in os.listdir(genre_folder_name):
    label = file_name.replace('class-', '')
    if label not in genres_to_ignore:   # ignore labels we're not interested in
        genres.append(label)
        with open(os.path.join(genre_folder_name, file_name), 'r') as f:
            i = 0
            for line in f:
                if int(line) == 1:
                    records[i].append(label)
                i += 1

concepts_file_name = os.path.join(target_folder, "genres.txt")
with open(concepts_file_name, 'w') as f:
    for label in genres:
        f.write("{0}\n".format(label))

features_file_name = os.path.join(target_folder, "features_{0}.csv".format(n_dims))
with open(features_file_name, 'w') as f:
    for record in records:  
        if(len(record) > n_dims):   # ignore records without labels
            f.write("{0}\n".format(",".join(record)))