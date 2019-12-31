import numpy as np
from sys import argv
from os import listdir, mkdir


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="Name of dataset to run on.")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="increase output verbosity")

exit()

ds = argv[1]
#seed = argv[ argv.index('--seed')+1 ] if '--seed' in argv else None
trial = argv[ argv.index('--trial')+1 ] if '--trial' in argv else 1
store_local = True
if 'Datasets' not in listdir('.'):
	mkdir('Datasets/')
if 'Results' not in listdir('.'):
	mkdir('Results/')

params = {}
params['trial'] = trial # Random seed
params['model_type'] = 'lr' # Base model
params['penalty'] = 'l2'
params['c_level'] = 'model' # Tuning at individual submodel or joint model level.


if ds.lower() == 'artificial':
	from data_artificial import Loader
	models = ['nb', 'lr', 'plr', 'plr_split', 'gec']
	if model not in models and 'rand' not in model:
		print("Model", model, "is not set up for use with artificial data.")

	params['n_p'] = int(argv[2])  # Number of partitions
	params['dim'] = int(argv[3])  # Partition size
	params['n_ol'] = int(argv[4]) # Number of overlapping features
	params['n_train'] = int(argv[5]) # Number of training examples
	#params['n_test']  = 2500 # Number of test examples
	params['n_test']  = 50

	#filename =  'fit-s%i-f%i-t_%i-%s%s.P'%(
	#			trial, params['n_p'], params['n_train'], m, trial_ext)


elif ds.lower() == 'newsgroups':
	from data_newsgroups import Loader
	models = ['nb', 'lr', 'plr']
	if model not in models and 'rand' not in model:
		print("Model", model, "is not set up for use with newgroups data.")
	''' Pair abbreviations:
		- 'ac' = ['alt.atheism', 'soc.religion.christian']
		- 'hb' = ['rec.sport.hockey', 'rec.sport.baseball']
		- 'wg' = ['comp.windows.x', 'comp.graphics']
		To extend, please edit newsgroup_data __init__ function to
			include your desired pair abbreviation.
	'''
	params['pair'] = argv[2]
	params['n_f'] = int(argv[3]) # Number of words to use as features
	params['n_train'] = int(argv[4]) # Number of training examples
	#params['n_test']  = 2000 # Number of test examples
	params['n_test']  = 50

elif ds.lower() == 'mnist':
	from data_mnist import Loader
	models = ['nb', 'lr', 'plr', 'plr_split', 'gec']
	if model not in models:
		print("Model", model, "is not set up for use with mnist data.")
	params['pair'] = sorted([int(argv[2]), int(argv[3])])
	params['scheme'] = argv[4] # Partitioning method
		# Note: scheme should be one of: focal, diagonal, grid
	params['n_train'] = int(argv[5]) # Number of training examples
	params['n_test'] = None # Number of test examples pre-determined

elif ds.lower() == 'medifor':
	from data_medifor import Loader
	print("Sorry, MediFor data cannot yet be shared...")
	exit()
	models = ['nb', 'lr', 'plr', 'rand']
	if model not in models and 'rand' not in model:
		print("Model", model, "is not set up for use with MediFor data.")
	params['dataset'] = argv[2]
	params['train_frac'] = float(argv[3]) # Fraction of data to use
											# as training set

else:
	print("Error: dataset", ds, "not recognized.")
	print("Please choose one of artificial, newsgroups, mnist, or medifor, or define new dataset structure.")
	exit()

loader = Loader(params)
loader.create(params['n_train'], params['n_test'], overwrite=False, verbose=False)

loader.load() # load existing data (and create overlap for artificial)
loader.make_partitions() # partition data into proper sets
X, Y = loader.X, loader.Y

print(X['train'].keys())
exit()

from controller import *

### Train
controller.train_model(params)

### Eval
controller.eval_model(params)


### Plot


exit()

np.random.seed(trial)






















