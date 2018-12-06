#!/bin/bash
# 1st argument: name of template


source activate tensorflow-CS
python ../Utilities/grid_search.py template.cfg $1
source deactivate tensorflow-CS

