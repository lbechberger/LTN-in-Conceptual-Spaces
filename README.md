# LTN-in-Conceptual-Spaces
Applying Logic Tensor Networks in Conceptual Spaces

Logic Tensor Networks ([Paper](https://arxiv.org/abs/1606.04422), [Code](https://www.dropbox.com/sh/502aq9u537lrmbv/AABuiKT4cKN-P3J7RIXd3cQ9a?dl=0)) provide a way of connecting logical rules with a feature space representation.

In this repository, we use LTNs to learn concepts in a conceptual space.

Copyright of "logictensornetworks.py" is retained by Luciano Serafini and Artur d'Avila Garcez. The files in `data/Schockaert/` were created based on data downloadable from (https://www.cs.cf.ac.uk/semanticspaces/) and reported in Joaquín Derrac and Steven Schockaert. Inducing semantic relations from conceptual spaces: a data-driven approach to commonsense reasoning, Artificial Intelligence, vol. 228, pages 66-94, 2015.


## Requirements

Our code was written in Python 2.7 and has dependencies on tensorflow, pylab, and matplotlib. Using `pip install -r requirements.txt` should install all necessary requirements.

## Config files

The configuration files contain all LTN hyperparameters as well as the setup of the concrete experiment. This allows us to keep the actual `run_ltn.py` quite general.
Configuration files look as follows:
```
[ltn-default]
# number of receptive fields per predicate; default: 5
ltn_layers = 1
# factor to which large weights are penalized; default: 0.0000001
ltn_smooth_factor = 1e-10
# appropriate t-conorm is used to compute disjunction of literals within clauses; options: 'product', 'yager2', 'luk', 'goedel'; default: 'product'
ltn_tnorm = luk
# aggregation across data points when computing validity of a clause; options: 'product', 'mean', 'gmean', 'hmean', 'min'; default: 'min'         
ltn_aggregator = min
# optimizing algorithm to use; options: 'ftrl', 'gd', 'ada', 'rmsprop'; default: 'gd' 
ltn_optimizer = rmsprop
# aggregate over clauses to define overall satisfiability of KB; options: 'min', 'mean', 'hmean', 'wmean'; default: 'min'   
ltn_clauses_aggregator = hmean 
# penalty for predicates that are true everywhere; default: 1e-6
ltn_positive_fact_penalty = 1e-5
# initialization of the u vector (determining how close to 0 and 1 the membership values can get); default: 5.0
ltn_norm_of_u = 5.0

[simple]
# only 4 concepts (banana, pear, orange, lemon) with clean data & no rules
concepts_file = data/fruit_space/concepts_simple.txt
features_file = data/fruit_space/features_simple.csv
rules_file = data/fruit_space/rules_simple.txt
num_dimensions = 3
training_percentage = 0.5
max_iter = 1000
```
The section `ltn-default` sets the default LTN hyperparameters. Individual sections like `simple` define the files to use, the size of the space, the percentage of feature vectors to use for training, and the maximal number of iterations for the optimization algorithm. They can also override the LTN hyperparameters set in the `ltn-default` section by re-defining them.

## Data format

The input to the LTN training algorithm is given in three separate files.

### conepts_file

This is a regular text file which simply lists the different concepts that one would like to learn. Each concept is listed in a single line. Note that all the concepts used in the features_file must also appear in the concepts_file.

### features_file

This is a csv file without a header where colums are separated by commas. The first `n` colums contain the vector/point and all remaining columns contain the concept labels. There must be at least one label per data point, but there can be arbitrary many. Different data points can have different numbers of labels. All labels used in this file have to be defined in the concepts_file. Moreover, the number of colums used for the vector needs to be equivalent to the `num_dimensions` in the config file.

The following two lines illustrate how the content of a features_file should look like:
```
0.338651517434,0.108252320347,0.240840991761,banana
0.658849789294,0.740900463574,0.289255306485,GrannySmith,apple
```

### rules_file
This is a regular text file that contains rules that should be taken into account when learning the concepts. Each rule is written in a separate line and can only involve concepts defined in the concepts_file.
Currently, the following types of rules are supported:
* `FirstConcept DIFFERENT SecondConcept` ensures that there is only little overlap between the two given concepts.
* `FirstConcept IMPLIES SecondConcept` ensures that the first concept is a subset of the second concept.

## Running the program

When you've set up your input files and your configuration file, you can execute the program as follows:
```
python ltn_code/run_ltn.py configFile.cfg configName
```
Here, `configFile.cfg` is the name of your configuration file and `configName` is the name of the configuration within that file that you would like to use (i.e., the specific experiment you would like to run). After initializing everything (which takes a couple of seconds), the script will display for each iteration the current satisfiability of the given constraints (both labeled data points and rules). In the end, the part of the data set that was not used for training is used to test the classification accuracy of the trained networks and some evaluation metrics are printed out. Moreover, if your data has two or three dimensions, colored scatter plots are generated to illustrate the location of the learned concepts in the overall space.
