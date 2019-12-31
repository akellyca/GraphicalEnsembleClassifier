### =================================================== ###
'''
### DESCRIPTION:
File for generating local datasets.
Generates datasets with parameters as specified in the file
	./config_[dataset].py

Stores resulting datasets in:
	./Datasets/[dataset]/*

### USAGE:
	generate_data.py [dataset]

Current version supports datasets:
	- 'artificial'
	- 'newsgroups'
	- 'mnist'
Code also exists for 'MediFor', but the dataset is currently
		unavailable for public release.
If you wish to expand support to more datasets, please add
corresponding data_[dataset].py file with Loader class to
use this file to generate. Also add a config_[dataset].py and
a case for your dataset in helper.py 'parse_params' function.

'''
### =================================================== ###

### Python module imports
import numpy as np
from sys import argv

### Local file imports
import helper


### Parse input arguments and initialize directories
dataset = argv[1].lower()
loader_module = __import__("data_"+dataset)
		
helper.init_direc('Datasets/'+dataset)
params = helper.parse_config(dataset)

### Generate desired datasets
for seed in params['seeds']:
	seed = int(seed)
	params['seed'] = seed
	
	if dataset=='artificial':
		params['n_ol'] = 0
		loader = loader_module.Loader(params)
		loader.generate()
		loader.make_partitions(n_ol=0)
		loader.save_data(overwrite=params['overwrite'])
		for ol in params['overlaps']:
			params['n_ol'] = ol
			loader.make_partitions(n_ol=ol)
			loader.save_data(overwrite=params['overwrite'])
	else:
		loader = loader_module.Loader(params)
		loader.generate()
		loader.make_partitions() # partition data into proper sets
		loader.save_data(overwrite=params['overwrite'])













