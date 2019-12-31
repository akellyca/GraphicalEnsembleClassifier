from os import listdir, mkdir
import numpy as np
import model_config
import pickle

config = None

def init_config(dataset):
	# Set up what dataset we are using
	global config
	config = __import__("config_"+dataset)


def load_dataset(file):
	try:
		with open(file, 'rb') as fh:
			loaded_data = pickle.load(fh)
	except:
		print("Dataset", file, '''does not exist.
					Please create before attempting to read.''')
		exit()
	return loaded_data

def load_fits(filename):
	data = []
	with open(filename, 'rb') as fh:
		while True:
			try:
				data.append( pickle.load(fh) )
			except:
				break
	if len(data)!=3:
		print("Fit file", filename, '''has an improper number of
				elements. Should contain: params, fit, trained.
				Please update before attempting to train.''')
		exit()
	else:
		params, fit, trained_sms = data
		return params, fit, trained_sms

def load_results(files, direc):
	results = {}
	models = []
	train_ns = []

	for f in files:
		file = f.split("-")
		seed = int(file[-3][4:])
		ntrain = int(file[-2][5:])
		model = file[-1][:-2]
		with open(direc+f, 'rb') as fh:
			res_params = pickle.load(fh)
			acc = pickle.load(fh)
			loss = pickle.load(fh)
			roc = pickle.load(fh)
		if 'rand' in model:
			model = '_'.join(s for s in model.split('_')[:-1])
		if model not in results.keys():
			results[model] = {}
		if ntrain not in results[model].keys():
			results[model][ntrain] = {'acc':[], 'loss':[], 'roc':[]}
		results[model][ntrain]['acc'].append(acc)
		results[model][ntrain]['loss'].append(loss)
		results[model][ntrain]['roc'].append(roc)
		if ntrain not in train_ns:
			train_ns.append(ntrain)
		if model not in models:
			models.append(model)

	for model in models:
		for ntrain in results[model].keys():
			results[model][ntrain]['acc'] = np.mean(results[model][ntrain]['acc'])
			results[model][ntrain]['loss'] = np.mean(results[model][ntrain]['loss'])

	train_ns = sorted(train_ns)
	models = order_models(models)
	return results, models, sorted(train_ns)

def parse_config(dataset):
	init_config(dataset)
	params = config.set_generation_params()
	return params

def substitute_plot_name(params):
	return config.save_template.substitute(params)

def parse_plot_params(ds_args):
	parsed = config.parse_ds_args(ds_args)
	return parsed

def get_plot_files(file_base, direc, with_rand):
	files = []
	for f in listdir(direc):
		if file_base not in f:
			continue
		if not with_rand and 'rand' in f:
			continue
		files.append(f)
	return files


def get_model_config():
	params = {}
	params['base_model'] = model_config.base_model
	params['penalty'] = model_config.penalty
	params['Cs'] = model_config.Cs
	params['n_folds'] = model_config.n_folds
	params['w_sms'] = model_config.w_sms
	return params


def parse_args(args):
	dataset = args[1]
	model = args[2]
	seed = int(args[3])
	n_train = int(args[4])
	parsed = {'dataset':dataset, 'model':model, 'seed':seed, 'n_train':n_train}
	
	ds_args = args[5:]
	ds_params = config.parse_ds_args(ds_args)
	ds_params['seed'] = seed
	parsed['file_base'] = config.file_template.substitute(ds_params)
	parsed['train_file'] = config.direc+parsed['file_base']+'-train.P'
	parsed['test_file'] = config.direc+parsed['file_base']+'-test.P'
	parsed.update(ds_params)
	return parsed


def set_rand_splits(m, sms, X):
	splits = {}

	nf = np.shape(X)[1]
	n_gps = int(m.split('_')[1][:-3])
	trial = int(m.split('_')[2][5:])
	gp_size = nf//n_gps

	rand_split_seed = nf + n_gps + trial
	splitter = np.random.RandomState(rand_split_seed)
	inds = [i for i in range(nf)]
	splitter.shuffle(inds)

	fs = [inds[gp_size*n:gp_size*(n+1)] for n in range(n_gps)]

	for i in range(n_gps):
		sm = 'rand_%igps_%i'%(n_gps,i)
		splits[sm] = X[:,fs[i]]

	return splits

def store_fit(objs_to_store, filename):

	with open(filename, 'wb') as fh:
		for obj in objs_to_store:
			pickle.dump(obj, fh)

	return

def init_direc(direc):
	### Given a directory path, makes that path starting from ./
		# (if the full path does not already exist)
	subs = direc.split('/')
	for i in range(len(subs)):
		if subs[i]=='' and i==len(subs)-1:
			break # accidentally put a trailing '/'
		curr_direc = './'+'/'.join(subs[:i])
		if subs[i].lower() not in [l.lower() for l in listdir(curr_direc)]:
			mkdir(curr_direc+'/'+subs[i])
	return

def get_sms(m, all_sms):
	sm_pairs = []

	if m in ['lr', 'nb']:
		sm_pairs = ['all']
	elif m == 'plr':
		for sm in all_sms:
			if sm[0]=='X' and sm[1:].isdigit():
				sm_pairs.append(sm)
	elif m=='plr_split':
		for sm in all_sms:
			if 'split' in sm:
				sm_pairs.append(sm)
	elif m=='gec':
		for sm in all_sms:
			spl = sm[1:].split('_')
			if sm[0]=='X' and sm[1:].isdigit():
				sm_pairs.append(sm)
			elif sm[0]=='X' and all([spl[i].isdigit() for i in range(len(spl))]):
				sm_pairs.append(sm)
	elif 'rand' in m:
		n_gps = int(m.split('_')[1][:-3])
		sm_pairs = ['rand_%igps_%i'%(n_gps,i) for i in range(n_gps)]
		#sm_pairs = [sm for sm in all_sms if 'rand_%igps'%n_gps in sm]


	return sm_pairs


def order_models(unsorted):
	# DESIRED ORDER:
		# NB, Split_n, ..., Split_1, LR, PLR, PLR_Split, GEC

	unsorted = [m.lower() for m in unsorted]
	models = []

	if "nb" in unsorted:
		models.append(["nb", "NB"])

	rands = []
	for m in unsorted:
		if m[:4]=="rand":
			n_gps = ''
			for c in m[5:]:
				if c.isdigit():
					n_gps = n_gps + c
				else:
					break
			rands.append(int(n_gps))

	for s in sorted(rands, reverse=True):
		if s==1:
			models.append(['rand_1gps', '1 group'])
		else:
			models.append(['rand_%igps'%s, str(s)+' groups'])

	if "lr" in unsorted:
		models.append(['lr', "LR"])
	if "plr" in unsorted:
		models.append(['plr', "PLR"])
	if "plr_split" in unsorted:
		models.append(['plr_split', "PLR Split"])
	if "gec" in unsorted:
		models.append(['gec', "GEC"])

	return models

#import argparse
# def _make_parser():
# 	parser = argparse.ArgumentParser()
# 	parser.add_argument("dataset", help="Name of dataset to run on.")
# 	parser.add_argument("n_train", help="Number of training examples.")
# 	#parser.add_argument("train_file",
# 	#		help='''Name of the specific training data file.
# 	#			Will search in ./Datasets/[dataset]/* ''')
# 	parser.add_argument("-v", "--verbose", 
# 			help="Verbose: print status messages.")
# 	return parser




