### 20-Newsgroup data parameters ###

### DATASET GENERATION PARAMETERS
pair = ['alt.atheism', 'soc.religion.christian']
shorthand = 'ac'
nf = 1000 # Number of words to use as features

n_train = 1000 # total train points to generate
n_test = 500 # total test points to generate
n_trials = 1 # number of seeds to generate for
overwrite = False # Overwrite existing trials

# Load dataset-specific arguments (for train/test)
def parse_ds_args(args):
	parsed = {}
	#parsed['shorthand'] = args[0]
	parsed['pair'] = args[0]
	parsed['nf'] = int(args[1])
	parsed['n_trained'] = int(args[2])
	return parsed

# Load generation parameters (for generate_data)
def set_generation_params():
	params = {}
	params['pair'] = pair
	params['shorthand'] = shorthand
	params['n_f'] = nf
	params['n_trained'] = n_train
	params['n_test'] = n_test
	params['seeds'] = [int(x) for x in range(n_trials)]
	params['overwrite'] = overwrite
	return params

### DEFINE SAVE LOCATION AND FILES
direc = 'Datasets/Newsgroups/'
import string
file_template = string.Template("$pair-f$nf-train$n_trained-seed$seed")
#file_params = {'pair':shorthand, 'nf':nf, 'n_trained':n_train, 'seed':0}
save_template = string.Template("$shorthand-f$nf-train$n_train")


