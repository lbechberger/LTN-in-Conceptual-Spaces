# -*- coding: utf-8 -*-
"""
Analyzes the given data set and prints out its characteristics.

1st argument: config file, 2nd argument: config_name

Created on Thu Feb  8 09:33:14 2018

@author: lbechberger
"""

import util, sys

config = util.parse_config_file(sys.argv[1], sys.argv[2])
util.data_set_characteristics(config["training_vectors"], config["validation_vectors"], config["test_vectors"], config["concepts"])