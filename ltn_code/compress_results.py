# -*- coding: utf-8 -*-
"""
Replaces for each configuration the results of individual runs by their overall average.

Created on Wed Feb 21 14:37:49 2018

@author: lbechberger
"""

import sys
import numpy as np

input_file_name = sys.argv[1]
output_file_name = sys.argv[2]

with open(input_file_name, 'r') as in_file:
    with open(output_file_name, 'w') as out_file:
        
        data_points = {}        

        def try_add(config, data_set, vector):
            if config not in data_points.keys():
                data_points[config] = {}
            if data_set not in data_points[config].keys():
                data_points[config][data_set] = []
            data_points[config][data_set].append(vector)
            
        
        for line in in_file:
            if (line.startswith("config")):
                # first line - just copy (and add 'count' column)
                out_line = "{0};{1}\n".format(line.replace("\n", "").replace(",",";"), 'counter')
                out_file.write(out_line)
            else:
                # regular line --> add to dictionary
                tokens = line.replace(",\n", '').split(",")
                try_add(tokens[0], tokens[1], list(map(lambda x: float(x), tokens[2:])))
        
        for config, dictionary in data_points.items():
            for data_set, vectors in dictionary.items():
                array = np.array(vectors)
                averages = np.mean(array, axis=0)
                out_file.write("{0};{1};{2};{3}\n".format(config, data_set, ";".join(map(lambda x: str(x), averages)), len(array)))            