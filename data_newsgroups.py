import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from os import listdir, mkdir
from operator import itemgetter
from scipy.sparse import csr_matrix
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import pickle

class Loader:

	def __init__(self, params):
		self.pair = params['pair']
		self.shorthand = params['shorthand']
		self.nf = params['n_f']
		self.n_train = params['n_train']
		self.n_test = params['n_test']
		self.seed = params['seed']
		self.vocabulary = []

		self.params = params
		self.origX = {}
		self.X, self.Y = {}, {}


	def generate(self, verbose=False):
		''' Generate newsgroup dataset for given pair, nf, n_train, n_test.
			Stores data in filepath+filename.
			This function should only be accessed from generate_data.py
			Should only be used to generate a full dataset with
				the maximum desired number of training and test points.
		'''

		if self.seed!=None:
			np.random.seed(self.seed)

		ng_all = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'),
								categories=self.pair)

		if self.n_train+self.n_test>len(ng_all.target):
			print("AssertionError: asked for %i train and %i test but dataset only contains %i examples."%(
						self.n_train, self.n_test, len(ng_all.target)))
			exit()

		inds = [i for i in range(len(ng_all.data))]
		np.random.shuffle(inds)
		X_train = [ng_all.data[i] for i in inds[:self.n_train] ]
		Y_train = [ng_all.target[i] for i in inds[:self.n_train] ]
		X_test = [ng_all.data[i] for i in inds[-self.n_test:] ]
		Y_test = [ng_all.target[i] for i in inds[-self.n_test:] ]

		stop_words = set(stopwords.words('english'))
		stop_words.add('nt')
		table = str.maketrans('', '', string.punctuation)

		train_docs, train_wds = self._clean(X_train, stop_words, table)
		test_docs, test_wds = self._clean(X_test, stop_words, table)

		counts = {}
		for w in train_wds:
			if w not in counts.keys():
				counts[w] = 0
			counts[w]+=1

		if self.nf >= len(counts.keys()):
			self.nf = len(counts.keys())
		assert len(counts.keys()) >= self.nf
		
		top = sorted(counts.items(), key=itemgetter(1), reverse=True)[:self.nf]
		self.vocabulary = [t[0] for t in top]
		self.params['vocabulary'] = self.vocabulary

		ng_train = csr_matrix( self._restrict_features(train_docs) )
		ng_test = csr_matrix( self._restrict_features(test_docs) )

		self.origX = { "train": ng_train[:,:], "test": ng_test[:,:] }
		self.Y = {"train":Y_train, "test": Y_test}

		return self.origX, self.Y

	def make_partitions(self):
		labels = pos_tag(self.vocabulary)
		self.pos_set = sorted(list(set([l[1] for l in labels])))
		pos_map = {pos:[] for pos in self.pos_set}
		inds_map = {pos:[] for pos in self.pos_set}
		for l in labels:
			pos_map[l[1]].append(l[0])
			inds_map[l[1]].append(self.vocabulary.index(l[0]))

		self.X = {'train':{}, 'test':{}}
		for t in ['train', 'test']:
			ctr = 1
			self.X[t]['all'] = self.origX[t][:,:]
			for i in range(len(self.pos_set)):
				pos = self.pos_set[i-1]
				label = 'X'+str(ctr)
				self.X[t][label] = self.origX[t][:,inds_map[pos]]
				ctr+=1
		return self.X, self.Y

	def save_data(self, file_base=None, direc=None, overwrite=True, verbose=True):
		''' Stores data into pickle file in Datasets/NewsgroupSets/ directory.
			dataset = { 'X': {'train':[data], 'test':[labels]},
						'Y': {'train':[data], 'test':[labels]},
						'params': parameter set (for future use)
					}
			(* Included in train file only.)
		'''

		### Specify directory and filenames to store in
		self.direc = direc if direc!=None else 'Datasets/Newsgroups/'
		self.file_base = file_base if file_base!=None else '%s-f%i-train%s-seed%i'%(
							self.shorthand,self.nf,self.n_train,self.seed)
		self.train_file = self.file_base+'-train.P'
		self.test_file = self.file_base+'-test.P'
		self.params['direc'] = self.direc
		self.params['filename'] = self.file_base
		self.params['train_file'] = self.train_file
		self.params['test_file'] = self.test_file

		if self.direc=='Datasets/Newsgroups' and 'Newsgroups' not in listdir('Datasets/'):
			mkdir('Datasets/Newsgroups/')

		### Split data into train and test sets
		train_dataset = {'X': self.X['train'], 'Y': self.Y['train'],
						'params': self.params}
		test_dataset = {'X': self.X['test'], 'Y': self.Y['test'],
						'params': self.params}

		### Check for existing files, if necessary
		if not overwrite:
			if self.train_file in listdir(self.direc):
				if verbose:
					print("Training data file %s already exists!"%(self.direc+self.train_file))
					print("Please use override=True or delete the above file to overwrite.")
				return 0
			if self.test_file in listdir(self.direc):
				if verbose:
					print("Test data file %s already exists!"%(self.direc+self.test_file))
					print("Please use override=True or delete the above file to overwrite.")
				return 0
		
		### Store generated data
		with open(self.direc+self.train_file, 'wb') as fh:
			pickle.dump(train_dataset, fh)
		with open(self.direc+self.test_file, 'wb') as fh:
			pickle.dump(test_dataset, fh)
		return 1

	### ============================================================###
		### Auxiliary functions ###
	### ============================================================###


	def _clean(self, data, stop_words, table):
		tokens = [[w.lower() for w in word_tokenize(ex)] for ex in data]
		stripped = [[w.translate(table) for w in ex ] for ex in tokens]
		words = [[w for w in ex if w.isalpha()] for ex in stripped]
		
		documents = [[w for w in ex if w not in stop_words] for ex in words]
		words = [w for ex in documents for w in ex]
		return documents, words

	def _restrict_features(self, data):
		limited = []
		for ex in data:
			transformed = []
			for w in self.vocabulary:
				transformed.append(int(w in ex))
			limited.append(transformed)
		return limited



















