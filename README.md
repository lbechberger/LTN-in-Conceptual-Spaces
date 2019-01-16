# LTN-in-Conceptual-Spaces
Applying Logic Tensor Networks in Conceptual Spaces

Logic Tensor Networks ([Paper](https://arxiv.org/abs/1606.04422), [Code](https://www.dropbox.com/sh/502aq9u537lrmbv/AABuiKT4cKN-P3J7RIXd3cQ9a?dl=0)) provide a way of connecting logical rules with a feature space representation.

In this repository, we use LTNs to learn concepts in a conceptual space.

Copyright of "logictensornetworks.py" is retained by Luciano Serafini and Artur d'Avila Garcez. The files in `data/Schockaert/` were created based on data downloadable from (https://www.cs.cf.ac.uk/semanticspaces/) and reported in Joaquín Derrac and Steven Schockaert. Inducing semantic relations from conceptual spaces: a data-driven approach to commonsense reasoning, Artificial Intelligence, vol. 228, pages 66-94, 2015.


## Requirements

Our code was written in Python 2.7 and has dependencies on tensorflow (version 1.4), pylab, and matplotlib. Using `pip install -r requirements.txt` should install all necessary requirements. Alternatively, the script `makeCondaTensorflow.sge` creates a conda environment with all necessary libraries.

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
features_folder = data/fruit_space/features_simple/
rules_file = data/fruit_space/rules_simple.txt
num_dimensions = 3
max_iter = 1000
```
The section `ltn-default` sets the default LTN hyperparameters. Individual sections like `simple` define the files to use, the size of the space, and the maximal number of iterations for the optimization algorithm. They can also override the LTN hyperparameters set in the `ltn-default` section by re-defining them.

## Data format

The input to the LTN training algorithm is given in three types of files.

### conepts_file

This is a regular text file which simply lists the different concepts that one would like to learn. Each concept is listed in a single line. Note that all the concepts used in the features_file must also appear in the concepts_file.

### files in features_folder

In the folder specified by `features_folder` in the configuration file, three files named `training.csv`, `validation.csv`, and `test.csv` contain the training, validation, and test set, respectively.
Each of them is a csv file without a header where colums are separated by commas. The first `num_dimensions` colums contain the vector/point and all remaining columns contain the concept labels. There must be at least one label per data point, but there can be arbitrary many. Different data points can have different numbers of labels. All labels used in this file have to be defined in the concepts_file. Moreover, the number of colums used for the vector needs to be equivalent to the `num_dimensions` in the config file.

The following two lines illustrate how the content of such a file should look like:
```
0.338651517434,0.108252320347,0.240840991761,banana
0.658849789294,0.740900463574,0.289255306485,GrannySmith,apple
```

You can use the script `tools/split_data.py` to split your overall data set into these three parts automatically. 

### rules_file
This is a regular text file that contains rules that should be taken into account when learning the concepts. Each rule is written in a separate line and can only involve concepts defined in the concepts_file.
Currently, the following types of rules are supported:
* `FirstConcept DIFFERENT SecondConcept` ensures that there is only little overlap between the two given concepts.
* `FirstConcept IMPLIES SecondConcept` ensures that the first concept is a subset of the second concept.

## Running the Label Counting
The label counting baseline can be executed as follows:
```
python ltn_code/run_counting.py configFile.cfg configName
```
Here, `configFile.cfg` is the name of your configuration file and `configName` is the name of the configuration within that file that you would like to use. From this configuration, only the information about the data set is used (ignoring the feature vectors and taking only into account the label information), all LTN hyperparameters are ignored.

The program calculates the validity of 21 different rule types (1 rule type `A != B`, 4 rule types in the form of `A IMPLIES B` with negated and non-negated concepts, 8 rule types in the form of `(A AND B) IMPLIES C` with negated and non-negated concepts, and 8 rule types in the form of `A IMPLIES (B or C)` with negated and non-negated concepts) on the three data sets.
Afterwards, for a set of different thresholds (0.7, 0.8, 0.9, 0.95, and 0.99), it removes all rules that have an accuracy of less than this threshold on either the training or the validation set. For the remaining rules, the average and minimum accuracy on the test set are computed.

Information on the thresholds, on the average and minimum accuracy on the test set, and on the number of rules left are displayed on the console for each rule type individually. Moreover, different output files are created: An overall csv file in the `output` folder contains the same information as is displayed on the console. Moreover, for each combination of rule type and desired threshold, an individual csv file is created in the `output/rules` folder that contains a list of all the extracted rules under this condition along with their individual performance on training, validation, and test set.

## Running the classification baselines
We compare the classification performance to two simple baselines:
- **constant**: This baseline predicts for all labels and for all data points always a membership value of 0.5
- **distribution**: This baseline computes the frequency of the labels in the data set and uses these frequencies as a prediction for all data points.

You can execute the baseline script as follows:
```
python ltn_code/run_baseline_classifiers.py configFile.cfg configName
```
The evaluation results are displayed on the console and additionally written into a csv file in the `output` folder.

## Running the kNN

The kNN baseline can be executed as follows:
```
python ltn_code/run_knn.py configFile.cfg configName k
```
Here, `configFile.cfg` is the name of your configuration file and `configName` is the name of the configuration within that file that you would like to use. From this configuration, only the information about the data set is used, all LTN hyperparameters are ignored. Finally, the parameter `k` indicates the number of neighbors to use in the classification.

The program trains a kNN classifier on the training set and evaluates it on the validation set. In the end, some evaluation metrics for both the training and the validation set are printed out. In addition, all the evaluation information displayed on the console is also written into a csv file in the `output` folder.

## Running the LTN

When you've set up your input files and your configuration file, you can execute the LTN as follows:
```
python ltn_code/run_ltn.py configFile.cfg configName
```

The program trains an LTN on the training set and evaluates it on the validation and the test set. Evaluation results are stored in a csv file in the `output` folder.

It takes the following arguments:
- `configFile.cfg` is the name of your configuration file to use.
- `configName` is the name of the configuration within that file that you would like to use (i.e., the specific experiment you would like to run).
- `-t` or `--type`: If this flag is set, then the type of membership function to use is overwritten by the type given immediately after this flag.
- `-p` or `--plot`: If this flag is set and if your data has two or three dimensions, colored scatter plots are generated to illustrate the location of the learned concepts in the overall space.
- `-q` or `--quiet`: By default, the script prints the current satisfiability and the evaluation results on the terminal. If this flag is set, the output is reduced to a minimum.
- `-e` or `--early`: If this flag is set, then the LTN will stop training after reaching a satisfiability of `0.99`. Otherwise it will continue training until the number of epochs specified in the configuration is reached.
- `-r` or `--rules`: If this flag is set, then in each evaluation step the LTN tries to extract rules from the learned membership functions.


## Analyzing kNN and LTN results
We have programmed a script to automatically analyze which hyperparameter configurations perform best on the training or validation set. It is applicable to both the kNN and the LTN classifications.
This script can be executed as follows:
```
python tools/find_optimal_params.py input_csv_file data_set_to_analyze
```
Here, `input_csv_file` is the path to the csv output file created by either `run_knn.py` or by the `compress_results.py` script ran on the output of`run_ltn.py`, and `data_set_to_analyze` should be set to either `training` or `validation`. 

This script selects a subset of hyperparameter configurations and outputs them (together with their associated evaluation metric values) in a csv file located in the same directory as `input_csv_file`, using the same basic file name but with the `data_set_to_analyze` appended. So if you call the script like `python tools/find_optimal_params.py output/grid_search-LTN.csv validation`, the results will be stored in `output/grid_search-LTN_validation.csv`. 

The hyperparmeter configurations are selected in two ways:
* For each evaluation metric, the hyperparameter configuration achieving the optimal performance with respect to this metric is chosen.
* Moreover, the script searches for hyperparameter configurations that achieve a good performance with respect to multiple metrics (measured by belonging to the top 1,2,3, and 5 percentile). A configuration gets 4 points for being in the 1 percentile, 3 points for belonging to the 2 percentile, etc. for each of the metrics. The 1% of configurations with the highest total score are chosen (maximally 20 configurations in order to keep the resulting spreadsheet clean).

In addition to the selected configurations, the output file also contains a row **BEST** which contains the best value for each of the metrics that was achieved by *any* configuration, and a row **WORST** which records the worst observed values for each configuration.

## Plotting the performance distribution
In order to visualize the distribution of LTN performance with respect to a given metric, you can use the script `plot_performance_distribution.py`:
```
python tools/plot_performance_distribution.py input_csv_file metric
```
The parameter `input_csv_file` should contain one row for each of the different configurations. If configurations were run multiple times, this should thus be the averaged results (i.e., the output file of `compress_results.py`). The given `metric` must be one of the columns of this csv file. The script collects all the values achieved for this metric and creates some plots visualizing them: A histogram with 21 bins, a line graph, and a scatter plot. For the line graph and the scatter plot, the values are first sorted. The x-axis is just the index of the sorted lis and the y-axis gives the respective performance value. The script takes the following optional arguments:
- `-o` or `--output_folder`: The folder where the plots are stored. Defaults to `.`, i.e., the current working directory.
- `-d` or `--data_set`: Defines the data set to analyze. By default, `validation` is used.
- `-p` or `--percentage`: Percentage of data points to plot. Defaults to 10, which means that only the top 10% of the data points are plotted (in order to make differences more visible)
- `-m` or `--minimize`: Use this flag if the metric is to be minimized. Then, the bottom `-p` percent will be plotted. If this flag is not set, the top `-p` percent are plotted.
