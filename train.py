###===========================================###
'''
### DESCRIPTION:
Main function for training/fitting models.

Datasets to load are assumed to be in 
		./Datasets/[dataset]/[base_file]-train.P
	where [base_file] format can be updated in [dataset]_config.py
Trains a model for the provided setup parameters, using
model configuration from model_config.py.
Stores output fits in
./Fits/[dataset]/[base_model]/fit-[base_file]-train[n_train]-[model].P
e.g.: ./Fits/artificial/lr/fit-dim40-p2-ol10-seed0-train100-plr.P
	
### USAGE
train.py dataset model seed n_train [dataset_specific_params]
Accepted model parameters:
	- Artificial: dimension n_partitions n_overlap
		e.g. train.py artificial gec 0 1000 40 2 10
	- Newsgroups: pair n_features n_trained
		e.g. train.py newsgroups plr 1 500 ac 500 1000
	- MNIST: n1 n2 scheme
		e.g. train.py mnist lr 0 750 1 7 focal
Additional dataset-specific parameter options can be defined in
[dataset]_config.py and in corresponding data_[dataset].py __init__ and
helper.py functions (parse_params, parse_args).

'''
###===========================================###

### Python module imports
import numpy as np
from sys import argv

### Local file imports
import helper
from model import Model
import time

### Load proper parameters based on argv input
start = time.time()
helper.init_config(argv[1])
params = helper.parse_args(argv)
dataset = params['dataset']
model = params['model']
seed = params['seed']
n_train = params['n_train']

params.update( helper.get_model_config() )
fit_direc = 'Fits/'+dataset+'/'+params['base_model']+'/'
helper.init_direc(fit_direc)
params['fit_file'] = fit_direc+"fit-"+params['file_base']+"-train%i-%s.P"%(n_train,model)

### Load dataset, get submodels, and limit n_train appropriately
data = helper.load_dataset(params['train_file'])
assert np.shape(data['X']['all'])[0] >= n_train #Ensure enough data

all_sms = data['X'].keys()
sms = helper.get_sms(model, all_sms)
params['model'] = model
params['sms'] = sms

if 'rand' in model:
	data['X'].update( helper.set_rand_splits(model, sms, data['X']['all'][:n_train,:]) )
X = {sm: data['X'][sm][:n_train,:] if np.shape(data['X'][sm])!=(1,0)
		else np.array([]) for sm in sms}
y = data['Y'][:n_train]


M = Model(params)
if model=='nb':
	M.train_tune_nb(X, y)
else:
	M.train_sms(X,y)
	M.tune_sms(X,y)

p_y = sum(y)/len(y)
preds, rocs = M.predict(X, p_y)
train_acc = M.eval(preds, y)
loss = M.get_log_loss(preds, y)

store = [params, M.fit, M.trained]
helper.store_fit(store, params['fit_file'])
end = time.time()
print(model, n_train, " - Training accuracy: %.2f"%(100*train_acc))#, end-start)
#print("Training accuracy: %.2f"%(100*train_acc))
#print("Test log loss: %.2f"%(loss))








