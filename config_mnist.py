### MNIST data generation parameters
# For use with generate_data.py mnist
# Please ensure that the files mnist_test.csv and mnist_train.csv
# 	are present in the Datasets/Mnist folder. Download from online.


### MNIST data parameters ###

### DATASET GENERATION PARAMETERS
numbers = [1, 7]
scheme = 'focal' # Partitioning scheme: focal, diagonal, or grid.

n_train = 10000 # total train points to generate
# n_test not specified because train-test split is pre-defined
n_trials = 1 # number of seeds to generate for
overwrite = False # Overwrite existing trials

# Load dataset-specific arguments (for train/test)
def parse_ds_args(args):
	parsed = {}
	parsed['numbers'] = args[0]+"_"+args[1]
	parsed['scheme'] = args[2]
	return parsed

# Load generation parameters (for generate_data)
def set_generation_params():
	params = {}
	params['numbers'] = numbers
	params['scheme'] = scheme
	params['n_train'] = n_train
	params['seeds'] = [int(x) for x in range(n_trials)]
	params['overwrite'] = overwrite
	return params

### DEFINE SAVE LOCATION AND FILES
direc = 'Datasets/Mnist/'
import string
file_template = string.Template("$numbers-$scheme-seed$seed")
str_numbers = str(numbers[0])+str(numbers[1])
#file_params = {'numbers':str_numbers, 'scheme':scheme, 'seed':0}
save_template = string.Template("$numbers-$scheme")

