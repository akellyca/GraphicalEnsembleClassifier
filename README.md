# Graphical Ensemble Classifier Codebase

This repository contains the code associated to my Directed Research Project / Masters Thesis: "Exploiting Domain Structure using Hybrid Generative-Discriminative Models"

The associated project writeup can be found at:
http://ix.cs.uoregon.edu/~austenk/AustenDRP.pdf


This code is written and maintained by Austen Kelly.
If you have any questions or comments, please direct them to austenk@cs.uoregon.edu

Thank you!

## Code organization

This code is designed with separate modules for data generation, training, testing, and plotting. Due to the nature of the GEC model, the code requires many dataset-specific parameters, and hence the code has many dataset-specific files/functions. It contains support for a select set of datasets, but requires some work to add in additional dataset support.
It is organized so that the user can decide whether to run simulations individually (train.py/test.py) or launch a full set of trials (submit.sh).

The current version of this code supports evaluation on the following datasets:
1. Artificial
2. 20 Newsgroups
3. MNIST
4. Media Forensics (MediFor)*
(Note: The MediFor dataset is currently unavailable for public release, so it is omitted from this codebase.)

### generate_data.py
Run this file prior to attempting any training/testing. Generates datasets with proper format and partitions.

This file is used for generating local copies of data with user-specified numbers of examples and features. The specific dataset setups that it creates are determined by the parameters set in the corresponding config_[dataset].py file.

**Example Usage**
'' generate_data.py dataset''


### train.py
This file is used for isolated model training. User should provide the desired dataset, model, number of training examples, and corresponding dataset-specific parameters

**Usage**
'' train.py dataset model seed n_train [dataset_specific_parameters]''


### test.py
This file is used for isolated evaluation on test set of pre-trained models. Usage is the same as for train.py.

**Usage**
'' test.py dataset model seed n_train [dataset_specific_parameters]''

### submit.sh
This file is a bash file which may be used to submit a sequence of jobs to train/test on a given dataset. Recommended for use on a server, when wanting to run a full set of experiments at once.


### plot.py
A helper file for generating common plots (as seen in the paper), including: accuracy vs n_train, heatmap of model vs n_train, and accuracy relative to LR vs n_train.

## Adding New Dataset Support

In order to run this code with a dataset other than the ones provided, the following additions must be made to the codebase:
1. A file ''data_[ds_name].py'' with class ''Loader(params)'', for use with ''generate_data.py''. Must contain functions:
	- generate(): returns self.origX = {"train": train data, "test": test data } and self.Y = {"train": labels, "test": labels}
	- make_partitions(): returns self.X = {"train": partition: data, "test": partition: data} and self.Y (unchanged)
	- save_data(): stores data
2. A file ''config_[ds_name].py'', specifying:
	- dataset-specific arguments as needed for initializing the ''data_[ds_name].py'' Loader class.
	- a function ''parse_ds_args(args)'': takes in ds-specific args and returns a dictionary of ''{'param_name': value}''
	- a function ''set_generation_params()'': returns a dictionary of all of the dataset generation variables needed for the corresponding Loader class
	- a filepath and filename/savename template, with variables as set in the ''parse_ds_args'' function



