# -*- coding: utf-8 -*-
"""
Script for splitting up your data into training, validation, and test.

Call like this: python split_data.py path/to/file.csv train_percentage validate_percentage

Created on Wed Dec  6 09:40:16 2017

@author: lbechberger
"""

import sys, os, random

# fix random seed to ensure reproducibility
random.seed(42)

input_file_name = sys.argv[1]
train_percentage = float(sys.argv[2])
validate_percentage = float(sys.argv[3])

with open(input_file_name, 'r') as f:
    lines = f.read().splitlines()
random.shuffle(lines)

datasets = {}
first_cutoff = int(len(lines) * train_percentage)
second_cutoff = int(len(lines) * (train_percentage + validate_percentage))
datasets["training"] = lines[:first_cutoff]
datasets["validation"] = lines[first_cutoff:second_cutoff]
datasets["test"] = lines[second_cutoff:]

output_directory = input_file_name.replace('.csv', '')
os.makedirs(output_directory)

for part in ["training", "validation", "test"]:
    output_file_name = os.path.join(output_directory, "{0}.csv".format(part))
    with open(output_file_name, "w") as f:
        for line in datasets[part]:
            f.write("{0}\n".format(line))