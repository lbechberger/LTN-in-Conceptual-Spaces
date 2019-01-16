# -*- coding: utf-8 -*-
"""
Plots the performance distribution of the given data set for the given metric of interest.

Created on Tue Jan 15 15:07:00 2019

@author: lbechberger
"""

from matplotlib import pyplot as plt
import csv, argparse, os
import numpy as np

# parse command-line arguments
parser = argparse.ArgumentParser(description='creating some performance plots for easy visual analysis')
parser.add_argument('input_file', help = 'the input file containing the averaged results')
parser.add_argument('-o', '--output_folder', help = 'the folder to which the plots should be saved', default='.')
parser.add_argument('-d', '--data_set', help = 'the data set to analyze', default='validation')
parser.add_argument('-p', '--percentage', help = 'size of top percentile to look at', default = 0.1, type = float)
args = parser.parse_args()

to_minimize = ['one_error', 'coverage', 'ranking_loss', 'cross_entropy_loss']
to_maximize = ['average_precision', 'exact_match_prefix', 'min', 'mean']

all_metrics = to_minimize + to_maximize

# open csv file and read in all the information
reader = csv.DictReader(open(args.input_file, newline=''), delimiter=',')
all_values = {}
for metric in all_metrics:
    all_values[metric] = []
    
for line in reader:
    # only look at performance on data set of interest
    if line['data_set'] == args.data_set:
        for metric in all_metrics:
            all_values[metric].append(float(line[metric]))

# filter according to percentage given
filtered_values = {}
for metric in to_minimize:
    threshold = np.percentile(all_values[metric], int(100 * args.percentage))
    filtered_values[metric] = [val for val in all_values[metric] if val <= threshold]
    
for metric in to_maximize:
    threshold = np.percentile(all_values[metric], 100 - int(100 * args.percentage))
    filtered_values[metric] = [val for val in all_values[metric] if val >= threshold]
   
for metric in all_metrics:
    
    # histogram (on raw array)
    plt.hist(filtered_values[metric], bins=21)
    plt.title('histogram of {0} on {1} set'.format(metric, args.data_set))
    plt.savefig(os.path.join(args.output_folder, '{0}-{1}-hist.png'.format(metric, args.data_set)), bbox_inches='tight', dpi=200)
    plt.close()
    
    # line graph (on sorted array and range)
    x_coordinates = range(len(filtered_values[metric]))
    sorted_values = sorted(filtered_values[metric])
    plt.plot(x_coordinates, sorted_values)
    plt.title('distribution of {0} on {1} set'.format(metric, args.data_set))
    plt.savefig(os.path.join(args.output_folder, '{0}-{1}-line.png'.format(metric, args.data_set)), bbox_inches='tight', dpi=200)
    plt.close()
    
    # scatter graph (on sorted array and range)
    plt.scatter(x_coordinates, sorted_values)
    plt.title('distribution of {0} on {1} set'.format(metric, args.data_set))
    plt.savefig(os.path.join(args.output_folder, '{0}-{1}-scatter.png'.format(metric, args.data_set)), bbox_inches='tight', dpi=200)
    plt.close()
