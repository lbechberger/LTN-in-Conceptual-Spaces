# -*- coding: utf-8 -*-
"""
Small conversion tool to get small Movielens vectors into our internal LTN format.

@author: lbechberger
"""

import sys, csv

source_filename = sys.argv[1]
target_filename = sys.argv[2]

# read everything from the source file
data = []
current_movie = ''
current_record = []
with open(source_filename, 'r') as f:
    csvreader = csv.reader(f, delimiter=',', quotechar='"')
    for row in csvreader:
        if row[0] == 'movie':
            continue # ignore first line
        if row[0] == current_movie:
            # just add an additional label
            current_record.append(row[1])
        else:
            # store old line and start new one
            if len(current_record) > 0:
                data.append(current_record)
            current_record = row[2:]
            current_record.append(row[1])
            current_movie = row[0]
data.append(current_record)

# write it to the target file
with open(target_filename, 'w') as f:
    for record in data:
        f.write("{0}\n".format(",".join(record)))
