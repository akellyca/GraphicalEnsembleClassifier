import numpy as np
import matplotlib.pyplot as plt
import pickle
from seaborn import heatmap
from sys import argv
from os import listdir

import helper

'''
USAGE: 
	train.py dataset with_rand [dataset_specific_params]
	(where with_rand=0/1 and dataset params are as in train.py)
'''

### MANUALLY SET THESE FLAGS PRIOR TO USE ###
plot_acc = True
plot_heatmap = False
plot_comparison = False
save = True
show = True
ylim = False
xscale = 'linear'


###======================================###
### HELPER FUNCTIONS
###======================================###
def make_plot(xs, ys, line_labels, axis_labels, filename, xscale, ylim, save, show):
	colors = plt.cm.jet(np.linspace(0, 1, len(xs)))

	for i in range(len(xs)):
		plt.plot(xs[i], ys[i], color=colors[i], label=line_labels[i])

	plt.xscale(xscale)
	if ylim:
		plt.ylim(ylim[0], ylim[1])

	ncol=2 if len(xs)>5 else 1
	plt.legend(loc='lower right', fontsize='medium', ncol=2)
	plt.xlabel(axis_labels[0], fontsize='large')
	plt.ylabel(axis_labels[1], fontsize='large')
	if save:
		plt.savefig(filename)
	if show:
		plt.show()
	plt.clf()

def make_heatmap(hm, xs, ys, axis_labels, filename, save, show):
	ax = heatmap(hm, cmap='magma') #jet?

	plt.xlabel(axis_labels[0])
	plt.ylabel(axis_labels[1])
	plt.xticks(rotation=30, fontsize='x-small')
	plt.yticks(rotation=45, fontsize='x-small')
	ax.set_xticklabels(xs)
	ax.set_yticklabels(ys)

	if save:
		plt.savefig(filename)
	if show:
		plt.show()
	plt.clf()
###======================================###
###======================================###

### LOAD PARAMS
dataset = argv[1]
with_rand = argv[2]
helper.init_config(dataset)
ds_args = helper.parse_plot_params(argv[3:])

###
direc = "Plots/"+dataset+"/"
helper.init_direc(direc)
rand_ext = '-no_rand' if not with_rand else ''
scale_ext = '-log' if xscale=='log' else ''

params = helper.get_model_config()
result_direc = 'Results/'+dataset+"/"+params['base_model']+"/"
save_direc = 'Plots/'+dataset+"/"+params['base_model']+"/"
helper.init_direc(save_direc)

file_base = helper.substitute_plot_name(ds_args)
files = helper.get_plot_files(file_base, result_direc, with_rand)
results, models, train_ns = helper.load_results(files, result_direc)
if len(models)==0:
	print("No existing results for desired figure. Please run corresponding simulations first.")
	exit()

if plot_acc:
	filename = "acc_vs_ntrain-"+file_base+"%s%s.pdf"%(rand_ext,scale_ext)
	axis_labels = ["Number of Training Examples", "Accuracy"]
	xs, ys, line_labels = [], [], []
	for i in range(len(models)):
		m = models[i]
		present_ns = sorted(list(results[m[0]].keys()))
		accs = [results[m[0]][n]['acc'] for n in present_ns]
		xs.append(present_ns)
		ys.append(accs)
		line_labels.append(m[1])

	make_plot(xs, ys, line_labels, axis_labels, save_direc+filename,
					xscale, ylim, save, show)

if plot_heatmap:
	# NOTE: To make heatmap, need full set of results matching train_ns
		# for the base model (in order to compare properly/consistently)
	filename = "heatmap-"+file_base+".pdf"
	axis_labels = ["Number of Training Examples", "Model"]
	base = 'lr' if 'lr' in models else 'rand_1gps'
	hm = []
	ys = []
	for m in models:
		if m[0]!=base:
			hm.append([results[m[0]][n]/results[base][n] for n in train_ns])
			ys.append(m[1])
	make_heatmap(hm, train_ns, ys, axis_labels, save_direc+filename, save, show)


if plot_comparison:
	filename = "lr_comparison-"+file_base+"%s%s.pdf"%(rand_ext,scale_ext)
	axis_labels = ["Number of Training Examples", "Accuracy Rel. to LR"]
	base = 'lr' if 'lr' in models else 'rand_1gps'
	hm = []
	xs, ys, line_labels = [], [], line_labels
	for m in models:
		if m[0]!=base:
			xs.append(train_ns)
			ys.append([results[m[0]][n]/results[base][n] for n in train_ns])
			line_labels.append(m[1])

	make_plot(xs, ys, line_labels, axis_labels, save_direc+filename,
					xscale, ylim, save, show)


















