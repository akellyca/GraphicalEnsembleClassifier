import numpy as np
from scipy.stats import bernoulli
from scipy.sparse import csr_matrix
from os import listdir, mkdir
from glob import glob
import pickle

class Loader:

	def __init__(self, params):
		self.n_partitions = params['n_p']
		self.dim = params['dim']
		self.ol = params['n_ol']
		self.n_train = params['n_train'] 
		self.n_test = params['n_test']  
		self.seed = params['seed']

		self.full_dim = self.n_partitions * self.dim
		self.params = params
		self.origX = {}
		self.X, self.Y = {}, {}


	def generate(self, verbose=False):
		''' Generate new artificial dataset.
			Creates n_partitions independent partitions of size full_dim.
			This function should only be accessed from generate_data.py
			Note: Should only be used to generate a full dataset with
				the maximum desired number of training and test points.
		'''

		if self.seed!=None:
			np.random.seed(self.seed)

		data_len = self.n_train + self.n_test
		Y = bernoulli.rvs(0.5, size=data_len)

		mu1 = np.array([np.sqrt(0.5/self.dim) for i in range(self.n_partitions*self.dim)])
		mu = [list(-mu1), list(mu1)]
		sigma_y = [self._generate_cov() for y in [0,1]]

		X_hat = np.array(self._make_X_hat(mu, sigma_y, Y))
		X = np.array([ [v for j in range(self.n_partitions*self.dim)
					for v in self._to6binary(X_hat[i][j])]
				 	for i in range(data_len)]) # convert to binary

		self.full_dim = 6*self.full_dim

		self.origX = {'train':X[:self.n_train,:],
						'test':X[self.n_train:,:]}
		self.Y = {'train':Y[:self.n_train], 'test':Y[self.n_train:]}

		return self.origX, self.Y

	def make_partitions(self, n_ol=None):

		self.create_overlap(n_ol)

		keys = self.set_keys()
		splits = { 'train':{k:[] for k in keys},
						'test':{k:[] for k in keys}}

		for t in ['train', 'test']:
			splits[t]['all'] = self.X[t][:]

			for i in range(len(self.X[t])):
				for p in range(self.n_partitions):
					### X1, ..., X_n (PLR & GEC)
					base = 6*p*(self.dim - self.ol)
					splits[t]['X'+str(p+1)].append( self.X[t][i][base:base+6*self.dim] )

					### X1_2, ... , X(n-1)_n (GEC)
					if p<self.n_partitions-1 and self.ol!=0: #not first or last
						base = 6*(p+1)*(self.dim - self.ol)
						label = 'X'+str(p+1)+'_'+str(p+2)
						splits[t][label].append( self.X[t][i][base:base+6*self.ol] )
					
					### X1_split, ..., Xn_split (PLR_split)
					### This version splits overlap in half across sets
					label = 'X'+str(p+1)+'_split'
					if p==0:
						splits[t][label].append( self.X[t][i][0:6*self.dim-3*self.ol] )
					elif p==self.n_partitions-1:
						splits[t][label].append( self.X[t][i][len(self.X[t][i])-6*self.dim+3*self.ol:] )
					else:
						b = 6*p*(self.dim - self.ol) + 3*self.ol
						e = b + 6*(self.dim-self.ol)
						splits[t][label].append( self.X[t][i][b:e] )

			self.X[t] = {k:csr_matrix(splits[t][k]) for k in splits[t].keys()}

		return self.X, self.Y

	def create_overlap(self, n_ol=None):

		if n_ol!=None:
			self.ol = n_ol
		self.params['n_ol'] = self.ol

		if self.ol==0: # do not need to bother with this iteration
			self.X = {k:self.origX[k][:,:] for k in self.origX.keys()}
			return self.X
		
		X = {'train':[], 'test':[]}
		buffer_len = 6*(self.dim-self.ol)

		for t in ['train', 'test']:
			for i in range(len(self.origX[t])):
				temp = [] # buffer of shape (n_partitions, buffer_len)
				for p in range(self.n_partitions):
					temp2 = np.zeros(p*buffer_len)
					temp3 = self.origX[t][i][6*self.dim*p:6*self.dim*(p+1)] 
					temp4 = np.zeros((self.n_partitions-p-1)*buffer_len)
					temp.append(np.hstack([temp2,temp3,temp4]))
				X[t].append(np.amax(temp,axis=0))

		self.X = X
		return self.X

	def save_data(self, file_base=None, direc=None, overwrite=True, verbose=True):
		''' Stores data into pickle file in Datasets/Artificial/ directory
						(unless otherwise specified)
			dataset = { 'X': {'train':[data], 'test':[labels]},
						'Y': {'train':[data], 'test':[labels]}
					}
		'''

		### Specify directory and filenames to store in
		self.direc = direc if direc!=None else 'Datasets/Artificial/'
		self.file_base = file_base if file_base!=None else 'dim%i-p%i-ol%i-seed%i'%(
							self.dim,self.n_partitions,self.ol,self.seed)
		#self.file_base = file_base if file_base!=None else 'dim%i-p%i-seed%i'%(
		#					self.dim,self.n_partitions,self.seed)
		self.train_file = self.file_base+'-train.P'
		self.test_file = self.file_base+'-test.P'
		self.params['direc'] = self.direc
		self.params['filename'] = self.file_base
		self.params['train_file'] = self.train_file
		self.params['test_file'] = self.test_file

		if self.direc=='Datasets/Artificial' and 'Artificial' not in listdir('Datasets/'):
			mkdir('Datasets/Artificial/')

		### Split data into train and test sets
		train_dataset = {'X': self.X['train'], 'Y': self.Y['train'], 'params':self.params}
		test_dataset = {'X': self.X['test'], 'Y': self.Y['test'], 'params':self.params}

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

	def get_rand_splits(self, subset, trial, seed=None):
		self.rand_splits = {}
		files = listdir('./Splits')
		for file in glob('./Splits/d%i_ol%i_*splits.txt'%(self.dim,self.n_ol)):
			n_gps = int(''.join([ch for ch in file.split('_')[2] if ch.isnumeric()]))
			self.rand_splits['rand_%igps'%n_gps] = {}
			start_line = (n_gps+1)*(trial-1) #skip appropriate lines
			with open(file, 'r') as fh:
				for skip in range(start_line):
					fh.readline()
				for spl in range(n_gps):
					split = [int(i) for i in fh.readline().strip().split(',')]
					self.rand_splits['rand_%igps'%n_gps]['rand_%i'%spl] = self.X[subset]['all'][:,split]

		return self.rand_splits


	def make_rand_split(self, n_gps, seed=None):
		self.rand_splits = {'train':{}, 'test':{}}
		if seed!=None:
			print('seeding')
			np.random.seed(seed)
		inds = [i for i in range(self.X['train']['all'].shape[1])]
		np.random.shuffle(inds)
		group_size = len(inds)//n_gps

		for n in range(n_gps):
			fs = inds[group_size*n:group_size*(n+1)]
			print(len(fs))
			continue
			for t in ['train', 'test']:
				self.rand_splits[t]['rand_%i'%n] = self.X[t]['all'][:,fs]

		exit()
		return self.rand_splits


	def _generate_cov(self):
		V = np.random.uniform(-1,1, size=(self.full_dim,10))
		gram = V.dot(V.T)

		for i in range(self.full_dim):
			for j in range(self.full_dim):
				# check if they are in the same big box. if not, need to zero out
				if i//self.dim != j//self.dim:
					gram[i][j] = 0.0 #alpha*gram[i][j]

		min_eig = np.min(np.real(np.linalg.eigvals(gram)))
		if min_eig < 0: # corrects for small floating point errors
			gram -= 2*min_eig * np.eye(*np.shape(gram))

		min_eig = np.min(np.real(np.linalg.eigvals(gram)))
		if min_eig < 0:
			print("Warning: gram matrix has negative eigenvalues. Minimum: ", min_eig)

		return gram


	def _make_X_hat(self, mu, sigma, Y):
		X_hat = []
		neg = np.random.multivariate_normal(mu[0], sigma[0], size=len(Y)-sum(Y))
		pos = np.random.multivariate_normal(mu[1], sigma[1], size=sum(Y))
		i, j = 0, 0

		for k in range(len(Y)):
			if Y[k]==0:
				X_hat.append(neg[i])
				i += 1
			elif Y[k]==1:
				X_hat.append(pos[j])
				j += 1
		return X_hat


	def _to6binary(self,n):
		s = [0 if n>0 else 1] # sign bit
		n = abs(n)
		for e in [4,2,1]: # bits 2^2, 2^1, 2^0
			if n>=e:
				s.append(1)
				n -= e
			else:
				s.append(0)
		while len(s)<6: # last 2 bits: 2^-1, 2^-2
			n = 2*(n - int(n))
			s.append(int(n))
		return s


	def set_keys(self):
		k = ['all']
		k.extend(['X'+str(i+1) for i in range(self.n_partitions)])
		k.extend(['X'+str(i+1)+'_split' for i in range(self.n_partitions)])
		for i in range(self.n_partitions-1):
			k.append('X'+str(i+1)+'_'+str(i+2))
		return k

















