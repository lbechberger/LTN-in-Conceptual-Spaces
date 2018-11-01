# -*- coding: utf-8 -*-
"""
Automatically finds optimal parameters based on evaluation file.

Run as follows: python find_optimal_params.py file_to_analyze data_set_to_analyze

Created on Wed Oct 17 14:40:07 2018

@author: lbechberger
"""

import csv
import sys
import numpy
import heapq

input_filename = sys.argv[1]
data_set_to_analyze = sys.argv[2]

output_filename = "{0}_{1}.csv".format(input_filename.replace(".csv", ""), data_set_to_analyze)

reader = csv.DictReader(open(input_filename, newline=''), delimiter=',')

to_minimize = ['one_error', 'coverage', 'ranking_loss', 'cross_entropy_loss']
to_maximize = ['average_precision', 'exact_match_prefix', 'min', 'mean']

all_metrics = to_minimize + to_maximize

records = {}
overall_optimal_values = {}
overall_worst_values = {}
raw_values = {}

for metric in to_minimize:
    overall_optimal_values[metric] = [9999, []]
    overall_worst_values[metric] = [-9999, []]
    raw_values[metric] = []

for metric in to_maximize:
    overall_optimal_values[metric] = [-9999, []]
    overall_worst_values[metric] = [9999, []]
    raw_values[metric] = []

rowcounter = 0

for line in reader:
    # only look at performance on data set of interest, ignore rows with nans and 0 ranking loss (LTN collapsed)
    if line['data_set'] == data_set_to_analyze:
        rowcounter += 1        
        
        # store for later        
        records[line['config']] = line
        
        # look at all metrics that have to be minimized
        for metric in to_minimize:
            value = float(line[metric])
            raw_values[metric].append(value)
            
            # if better than previous optimal value: replace
            if value < overall_optimal_values[metric][0]:
                overall_optimal_values[metric][0] = value
                overall_optimal_values[metric][1] = [line['config']]
            # if equally good: append config to list
            elif value == overall_optimal_values[metric][0]:
                overall_optimal_values[metric][1].append(line['config'])
            
            # if worse than previous worst value: replace
            if value > overall_worst_values[metric][0]:
                overall_worst_values[metric][0] = value
                overall_worst_values[metric][1] = [line['config']]
            # if equally bad: append config to list
            elif value == overall_worst_values[metric][0]:
                overall_worst_values[metric][1].append(line['config'])
        
        # look at all metrics that have to be minimized
        for metric in to_maximize:
            value = float(line[metric])
            raw_values[metric].append(value)
            
            # if better than previous optimal value: replace
            if value > overall_optimal_values[metric][0]:
                overall_optimal_values[metric][0] = value
                overall_optimal_values[metric][1] = [line['config']]
            # if equally good: append config to list
            elif value == overall_optimal_values[metric][0]:
                overall_optimal_values[metric][1].append(line['config'])
            
            # if worse than previous worst value: replace
            if value < overall_worst_values[metric][0]:
                overall_worst_values[metric][0] = value
                overall_worst_values[metric][1] = [line['config']]
            # if equally bad: append config to list
            elif value == overall_worst_values[metric][0]:
                overall_worst_values[metric][1].append(line['config'])

print("Read {0} rows.".format(rowcounter))

header = ['config', 'data_set']
for metric in all_metrics:
    header.append(metric)
header.append('comment')
result = [header]

# compute overall best and worst 
optimum = ['BEST', data_set_to_analyze]
worst = ['WORST', data_set_to_analyze]
for metric in all_metrics:
    optimum.append(overall_optimal_values[metric][0])
    worst.append(overall_worst_values[metric][0])

optimum.append('nonexistent')
worst.append('nonexistent')
result.append(optimum)
result.append(worst)

# now look at the configurations that optimize the individual metrics
for metric in all_metrics:
    for opt_config in overall_optimal_values[metric][1]:
        result_record = [opt_config, data_set_to_analyze]
        for m2 in all_metrics:
            result_record.append(records[opt_config][m2])
        result_record.append(metric)
        result.append(result_record)

# joint optimal param setting: first look at the percentiles
percentiles = {}
for metric in all_metrics:
    if metric in to_minimize:
        percentages = [1,2,3,5]     # we want small values
    else:
        percentages = [99,98,97,95] # we want large values
    local_percentiles = []
    for percentage in percentages:
        local_percentiles.append(numpy.percentile(raw_values[metric], percentage))
    percentiles[metric] = local_percentiles

# now score all records according to these percentiles
priority_queue = []
for record_name, record in records.items():
    local_scores = []
    
    for metric in to_minimize:  # check if less than percentiles
        metric_score = 0
        for level in percentiles[metric]:
            if float(record[metric]) < level:
                metric_score += 1
        local_scores.append(metric_score)
        
    for metric in to_maximize:  # check if greater than percentiles
        metric_score = 0
        for level in percentiles[metric]:
            if float(record[metric]) > level:
                metric_score += 1
        local_scores.append(metric_score)
    
    # aggregate overall score    
    overall_score = sum(local_scores)
    
    # build modified record for later output
    modified_record = [record_name, data_set_to_analyze]
    for metric in all_metrics:
        modified_record.append(record[metric])
    # comment: overall score and individual scores
    modified_record.append("{0}: [{1}]".format(overall_score, '-'.join(map(lambda x: str(x), local_scores))))
    
    heapq.heappush(priority_queue, (-overall_score, modified_record))

# take the configurations with the highest score - either top 20 or highest 1% (based on smaller number)
limit = min(int(0.01 * len(records.keys())), 20)
for i in range(limit):
    result.append(heapq.heappop(priority_queue)[1])

# finally: compute correlations between the metrics
overall_value_table = []
for metric in all_metrics:
    overall_value_table.append(raw_values[metric])

corr_matrix = numpy.corrcoef(overall_value_table)

result.append([])
result.append(['correlations'])
for line in corr_matrix:
    result.append(line)

# write everyting into the output file
with open(output_filename, 'w', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    for record in result:
        writer.writerow(record)

