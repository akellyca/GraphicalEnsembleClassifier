### =================================================== ###
'''
### DESCRIPTION:
Main function for making test predictions models.

Pre-trained model fits to load are assumed to be in 
	./Fits/[dataset]/[base_model]/fit-[base_file]-train[n_train]-[model].P
	where [base_model] [base_file] format can be updated in
		[dataset]_config.py
Makes model predictions for the provided setup parameters, using fits
from corresponding train.py call and model configuration from model_config.py.
Stores results in
./Results/[dataset]/[base_model]/pred-[base_file]-train[n_train]-[model].P
e.g.: ./Results/artificial/lr/pred-dim40-p2-ol10-seed0-train100-plr.P


### USAGE:
test.py dataset model seed n_train [dataset_specific_params]
Accepted model parameters:
	- Artificial: dimension n_partitions n_overlap
		e.g. test.py artificial gec 0 1000 40 2 10
	- Newsgroups: pair n_features n_train
		e.g. test.py newsgroups plr 1 500 ac 500 1000
	- MNIST: n1 n2 scheme
		e.g. test.py mnist lr 0 750 1 7 focal

'''
### =================================================== ###

### Python module imports
import numpy as np
from sys import argv

### Local file imports
import helper
from model import Model


### Load proper parameters based on argv input
params = helper.parse_args(argv)
dataset = params['dataset']
model = params['model']
seed = params['seed']
n_train = params['n_train']

params.update( helper.get_model_config() )
fit_direc = 'Fits/'+dataset+'/'+params['base_model']+'/'
params['fit_file'] = fit_direc+"fit-"+params['file_base']+"-train%i-%s.P"%(n_train,model)
result_direc = 'Results/'+dataset+'/'+params['base_model']+'/'
helper.init_direc(result_direc)
params['result_file'] = result_direc+"pred-"+params['file_base']+"-train%i-%s.P"%(n_train,model)

### Load dataset, get submodels, and limit n_train appropriately
data = helper.load_dataset(params['test_file'])
all_sms = data['X'].keys()
sms = helper.get_sms(model, all_sms)
params['model'] = model
params['sms'] = sms

if 'rand' in model:
	data['X'].update( helper.set_rand_splits(model, sms, data['X']['all']) )
X = data['X']
y = data['Y']
# y_train: used for getting prior class probability
y_train = helper.load_dataset(params['train_file'])['Y'][:n_train]


M = Model(params)
p, M.fit, M.trained = helper.load_fits(params['fit_file'])

p_y = sum(y_train)/(len(y_train))
preds, rocs = M.predict(X, p_y)

acc = M.eval(preds, y)
loss = M.get_log_loss(preds, y)
roc = M.get_roc_curve(preds, y)

store = [params,acc,loss,roc,preds]
helper.store_fit(store,params['result_file'])
print(model, n_train, " - Test accuracy: %.2f"%(100*acc))
#print("Test accuracy: %.2f"%(100*acc))
#print("Test log loss: %.2f"%(loss))
































