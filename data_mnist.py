import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from os import listdir, mkdir
from operator import itemgetter
from scipy.sparse import csr_matrix
import pickle


class Loader:

	def __init__(self, params):
		self.numbers = params['numbers']
		self.scheme = params['scheme']# Partitioning method: focal/grid/diagonal
		self.n_train = params['n_train']
		# self.n_test = params['n_test'] # mnist train-test split is determined
		self.seed = params['seed']
		self.direc = 'Datasets/Mnist/'
		self.params = params
		self.origX = {}
		self.X, self.Y = {}, {}


	def generate(self):

		if self.seed!=None:
			np.random.seed(self.seed)

		X = {'train':[], 'test':[]}
		Y = {'train':[], 'test':[]}

		for t in ['train','test']:
			with open(self.direc+'mnist_'+t+'.csv', 'r') as fh:
				for line in fh:
					line = line.strip().split(',')
					label = int(line[0])
					if label not in self.numbers:
						continue
					X[t].append([int(x)/255 for x in line[1:]])
					Y[t].append(self.numbers.index(label))

		#self.n_train = len(Y['train'])
		assert self.n_train <= len(Y['train'])
		self.n_test = len(Y['test'])

		inds = [i for i in range(self.n_train)]
		np.random.shuffle(inds)
		X['train'] = [X['train'][i][:] for i in inds]
		Y['train'] = [Y['train'][i] for i in inds]

		self.origX = X
		self.Y = Y
		return self.origX, self.Y

	def make_partitions(self):
		if self.scheme == 'grid':
			self.sms = ['X1', 'X2', 'X3', 'X4', 'X1_2', 'X3_4', 'Xm_12_34']
			self.X = {'train':{sm:[] for sm in self.sms}, 'test':{sm:[] for sm in self.sms}}
		
			for t in ['train', 'test']:
				self.X[t]['all'] = csr_matrix( self.origX[t][:] )

				for i in range(len(self.Y[t])):
					img = self.origX[t][i][:]
					img = np.array(img).reshape(28,28)

					self.X[t]['X1'].append([v for x in img[:18] for v in x[:]])
					self.X[t]['X2'].append([v for x in img[9:] for v in x[:]])
					self.X[t]['X1_2'].append([v for x in img[9:18] for v in x[:]])
					self.X[t]['X3'].append([v for x in img[:] for v in x[:18]])
					self.X[t]['X4'].append([v for x in img[:] for v in x[9:]])
					self.X[t]['X3_4'].append([v for x in img[:] for v in x[9:18]])
					self.X[t]['Xm_12_34'].append([v for x in img[9:18] for v in x[9:18]])

		elif self.scheme == 'focal':
			self.sms = ['X1', 'X2', 'X3', 'X4', 'Xm', 'X1_m', 'X2_m', 'X3_m', 'X4_m']
			self.X = {'train':{sm:[] for sm in self.sms}, 'test':{sm:[] for sm in self.sms}}
		
			for t in ['train', 'test']:
				self.X[t]['all'] = csr_matrix( self.origX[t][:] )
				for i in range(len(self.Y[t])):
					img = self.origX[t][i][:]
					img = np.array(img).reshape(28,28)
					self.X[t]['X1'].append( [v for x in img[:14] for v in x[:14]] )
					self.X[t]['X2'].append( [v for x in img[:14] for v in x[14:]] )
					self.X[t]['X3'].append( [v for x in img[14:] for v in x[:14]] )
					self.X[t]['X4'].append( [v for x in img[14:] for v in x[14:]] )
					self.X[t]['Xm'].append( [v for x in img[7:21] for v in x[7:21]] )
					
					self.X[t]['X1_m'].append( [v for x in img[7:14] for v in x[7:14]] )
					self.X[t]['X2_m'].append( [v for x in img[7:14] for v in x[14:21]] )
					self.X[t]['X3_m'].append( [v for x in img[14:21] for v in x[7:14]] )
					self.X[t]['X4_m'].append( [v for x in img[14:21] for v in x[14:21]] )

		elif self.scheme == 'diagonal':
			self.sms = ['X1', 'X2', 'X3', 'X4', 'Xm', 'X1_m', 'X2_m', 'X3_m', 'X4_m']
			self.X = {'train':{sm:[] for sm in self.sms}, 'test':{sm:[] for sm in self.sms}}
		
			for t in ['train', 'test']:
				self.X[t]['all'] = csr_matrix( self.origX[t][:] )
				for i in range(len(self.Y[t])):
					img = self.origX[t][i][:]
					img = np.array(img).reshape(28,28)
					
					m1, m2, m3, m4 = [], [], [], []
					m1m, m2m, m3m, m4m = [], [], [], []
					for j in range(14):
						for k in range(28):
							if k in range(j,28-j-1):
								m1.append(img[j][k])
								m3.append(img[28-j-1][k])
								if j>=7:
									m1m.append(img[j][k])
									m3m.append(img[28-j-1][k])
							else:
								if j>k:
									m2.append(img[j][k])
									m2.append(img[28-j-1][k])
									if k>=7:
										m2m.append(img[j][k])
										m2m.append(img[28-j-1][k])
								else:
									m4.append(img[j][k])
									m4.append(img[28-j-1][k])
									if k<=21:
										m4m.append(img[j][k])
										m4m.append(img[28-j-1][k])
					
					self.X[t]['X1'].append(m1)
					self.X[t]['X2'].append(m2)
					self.X[t]['X3'].append(m3)
					self.X[t]['X4'].append(m4)
					self.X[t]['Xm'].append( [v for x in img[7:21] for v in x[7:21]] )
					self.X[t]['X1_m'].append(m1m)
					self.X[t]['X2_m'].append(m2m)
					self.X[t]['X3_m'].append(m3m)
					self.X[t]['X4_m'].append(m4m)

		else:
			print("Invalid partitoning scheme. Please choose one of grid, focal, or diagonal.")

		for t in ['train', 'test']:
			self.X[t]['all'] = csr_matrix( self.origX[t][:] )
			for sm in self.sms:
				self.X[t][sm] = csr_matrix(self.X[t][sm])
		
		return self.X, self.Y


	def _bin(self,v):
		return int(v>=127)


	def save_data(self, file_base=None, direc=None, overwrite=True, verbose=True):
		''' Stores data into pickle file in Datasets/MnistSets/ directory.
			dataset = { 'X': {'train':[data], 'test':[labels]},
						'Y': {'train':[data], 'test':[labels]},
						'params': setup parameters (for later use)
					}
		'''

		### Specify directory and filenames to store in
		self.direc = direc if direc!=None else 'Datasets/Mnist/'
		self.file_base = file_base if file_base!=None else '%i_%i-%s-seed%i'%(
							self.numbers[0],self.numbers[1],self.scheme,self.seed)
		self.train_file = self.file_base+'-train.P'
		self.test_file = self.file_base+'-test.P'
		self.params['direc'] = self.direc
		self.params['filename'] = self.file_base
		self.params['train_file'] = self.train_file
		self.params['test_file'] = self.test_file

		if self.direc=='Datasets/Mnist' and 'Mnist' not in listdir('Datasets/'):
			mkdir('Datasets/Mnist/')


		### Set up train and test sets to store
		train_dataset = {'X': self.X['train'], 'Y': self.Y['train'],
						'params':self.params}
		test_dataset = {'X': self.X['test'], 'Y': self.Y['test'],
						'params':self.params}

		if not overwrite:
			if self.train_file in listdir(self.direc):
				if verbose:
					print("Training data file %s already exists!"%(self.direc+self.train_file))
					print("Please use override=True or delete the above file to overwrite existing file.")
				return 0
			if self.test_file in listdir(self.direc):
				if verbose:
					print("Test data file %s already exists!"%(self.direc+self.test_file))
					print("Please use override=True or delete the above file to overwrite existing file.")
				return 0
		with open(self.direc+self.train_file, 'wb') as fh:
			pickle.dump(train_dataset, fh)
		with open(self.direc+self.test_file, 'wb') as fh:
			pickle.dump(test_dataset, fh)
		return 1


















