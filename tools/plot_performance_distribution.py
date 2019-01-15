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
parser.add_argument('metric', help = 'the name of the metric to investigate')
parser.add_argument('-o', '--output_folder', help = 'the folder to which the plots should be saved', default='.')
parser.add_argument('-d', '--data_set', help = 'the data set to analyze', default='validation')
parser.add_argument('-p', '--percentage', help = 'size of top percentile to look at', default = 0.5, type = int)
parser.add_argument('-m', '--minimize', action = 'store_true', help = 'set if metric is to be minimized')
args = parser.parse_args()


# open csv file and read in column of interest (store as array of values)
reader = csv.DictReader(open(args.input_file, newline=''), delimiter=',')
all_values = []
for line in reader:
    # only look at performance on data set of interest
    if line['data_set'] == args.data_set:
        all_values.append(float(line[args.metric]))

if args.minimize:
    threshold = np.percentile(all_values, args.percentage)
    filtered_values = [val for val in all_values if val <= threshold]
else:
    threshold = np.percentile(all_values, 100 - args.percentage)
    filtered_values = [val for val in all_values if val >= threshold]

# histogram (on raw array)
plt.hist(filtered_values, bins=21)
plt.title('histogram of {0} on {1} set'.format(args.metric, args.data_set))
plt.savefig(os.path.join(args.output_folder, '{0}-{1}-hist.png'.format(args.metric, args.data_set)), bbox_inches='tight', dpi=200)
plt.close()

# line graph (on sorted array and range)
x_coordinates = range(len(filtered_values))
sorted_values = sorted(filtered_values)
plt.plot(x_coordinates, sorted_values)
plt.title('distribution of {0} on {1} set'.format(args.metric, args.data_set))
plt.savefig(os.path.join(args.output_folder, '{0}-{1}-line.png'.format(args.metric, args.data_set)), bbox_inches='tight', dpi=200)
plt.close()

# scatter graph (on sorted array and range)
plt.scatter(x_coordinates, sorted_values)
plt.title('distribution of {0} on {1} set'.format(args.metric, args.data_set))
plt.savefig(os.path.join(args.output_folder, '{0}-{1}-scatter.png'.format(args.metric, args.data_set)), bbox_inches='tight', dpi=200)
plt.close()
