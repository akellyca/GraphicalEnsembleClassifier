### Artificial data parameters ###

### DATASET GENERATION PARAMETERS
n_partitions = 2
dim = 40
overlaps = [0, 10, 20, 30, 40]

n_train = 5000 # total train points to generate
n_test = 2500 # total test points to generate
n_trials = 1 # number of seeds to generate for
overwrite = False # Overwrite existing trials

# Load dataset-specific arguments (for train/test)
def parse_ds_args(args):
	parsed = {}
	parsed['dim'] = int(args[0])
	parsed['np'] = int(args[1])
	parsed['n_ol'] = int(args[2])
	return parsed

# Load generation parameters (for generate_data)
def set_generation_params():
	params = {}
	params['n_p'] = n_partitions
	params['dim'] = dim
	params['n_train'] = n_train
	params['n_test'] = n_test
	params['overlaps'] = overlaps
	params['seeds'] = [int(x) for x in range(n_trials)]
	params['overwrite'] = overwrite
	return params

### DEFINE SAVE LOCATION AND FILES
direc = 'Datasets/Artificial/'
import string
file_template = string.Template("dim$dim-p$np-ol$n_ol-seed$seed")
#file_params = {'dim':dim, 'np':n_partitions, 'n_ol':n_ol, 'seed':0}
save_template = string.Template("dim$dim-p$np-ol$n_ol")




















