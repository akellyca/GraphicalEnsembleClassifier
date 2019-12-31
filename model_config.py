### model_config ###
### Model tuning parameters

# Imports
from numpy import logspace

#####

# Type of model to fit each subset on (lr, svm, or forest)
base_model = 'lr'

# Regularization:
	#lr: penalty=l1/l2
	#svm: kernel=poly/rbf
	#forest: n_estimators
penalty = 'l2'

# Space of regularization constants to search
Cs = logspace(-3,3,7)

# Number of folds for cross validation
n_folds = 5

# Whether or not to fit weights for combining submodels
	# Warning: no current implementation for w_sms=True
w_sms = False 


#####
### Parameters used for launching full set of simulations

# Which models to train/test
models = ['nb', 'lr', 'plr', 'gec', 'rand']

# Which numbers of random groups
	# (n_gps above max # of features will be automatically omitted)
	# Will be ignored if 'rand' is not in models
rand_n_gps = [1, 2, 3, 4, 8, 16, 32, 64, 128, 'all']
rand_trials = [1]

