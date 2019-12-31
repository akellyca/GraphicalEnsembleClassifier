import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from os import listdir
from operator import itemgetter
from scipy.sparse import csr_matrix
import xlrd
import pickle


class Medifor:
	def __init__(self, dataset):
		self.dataset = dataset
		self.filepath = "MediforSets/"+self.dataset+"/"
		if self.dataset not in listdir("MediforSets/"):
			print("Filepath to data does not exist:", self.filepath)
			exit()
		self.__load_all()

	def __load_all(self):
		self.data = {'X':[], 'Y':[]}
		self.partitions = None
		self.indicators = []
		self.img_ids = {} # {id:binary label}

		with open(self.filepath+'/targets/manipulation/reference.csv', 'r') as fh:
			i = 0
			fh.readline()
			for line in fh:
				line = line.strip().split(',')
				self.img_ids[line[0]] = int(line[1])
		self.indicators = sorted(listdir(self.filepath+"indicators"))
		self.indicators = [ind for ind in self.indicators if ind[0]!='.']

		predictions = { } # maps indicator -> image name -> score
		for ind in self.indicators:
			predictions[ind] = { }
			with open(self.filepath+"indicators/"+ind+'/reference.csv', 'r') as fh:
				fh.readline() # skip header
				for line in fh:
					line = line.strip().split(',')
					if line[0]=='' or line[0] not in self.img_ids.keys():
						continue # skip images not in known labels
					if line[1] == '':
						predictions[ind][line[0]] = None
						continue
					if line[1][0]=='e': # formatting so that they can be converted
						line[1] = '1'+line[1]	# to floats
					elif line[1][0]=='-' and line[1][1]=='e':
						line[1] = '-1'+line[1:]
					p = -float(line[1][1:]) if '-'==line[1][0] else float(line[1])
					predictions[ind][line[0]] = p
			for img in self.img_ids.keys(): # safety check
				if img not in predictions[ind].keys():
					predictions[ind][img] = None

		### remove keys which have too many empty predictions
		n = len(self.img_ids.keys())
		bad = []
		for ind in self.indicators:
			missing = [ x for x in predictions[ind].values() if x==None ]
			if len(predictions[ind].keys()) - len(missing) < 0.5*n:
				bad.append(ind)
				del predictions[ind]
		self.indicators = [k for k in predictions.keys()]

		### Default: No prediction -> 0
		for ind in predictions.keys():
			for image in predictions[ind].keys():
				if predictions[ind][image] == None:
					predictions[ind][image] = 0

		### Populate dataset (eliminating img id references)
		for img in self.img_ids.keys():
			ind_preds = [predictions[ind][img] for ind in self.indicators]
			self.data['X'].append(ind_preds)
			self.data['Y'].append(self.img_ids[img])
		self.data['X'] = {"all": np.array(self.data['X'])}
		self.data['Y'] = {"all": np.array(self.data['Y'])}
		return

	def init_folds(self, n_splits=5, shuffle=False):
		self.kf = KFold(n_splits=n_splits, shuffle=shuffle)

	def train_test_split(self, train_ind, test_ind, fold, verbose=False):
		self.train_ind = train_ind
		self.test_ind = test_ind
		self.train = {'X':{}, 'Y':{}}
		self.test = {'X':{}, 'Y':{}}

		### set train/test objects to corresponding fold
		#for sm in self.data['X'].keys():
		for sm in self.partitions.keys():
			self.train['X'][sm] = self.data['X'][sm][train_ind,:]
			self.test['X'][sm] = self.data['X'][sm][test_ind,:]
		self.train['Y']['all'] = self.data['Y']["all"][train_ind]
		self.test['Y']['all'] = self.data['Y']["all"][test_ind]

	def normalize_train_test(self):
		#for sm in self.train['X'].keys():
		for sm in self.partitions.keys():
			scaler = StandardScaler()
			scaler.fit(self.train['X'][sm])
			scaler.transform(self.train['X'][sm])
			scaler.transform(self.test['X'][sm])


	def load_partitions(self, filename="partition2.xlsx"):
		xls = xlrd.open_workbook(self.filepath+filename, on_demand=True)
		sheet = xls.sheet_by_index(0)
		data = np.array([[sheet.cell_value(r, c) for c in range(sheet.ncols)] for r in range(sheet.nrows)])
		#data = data.T
		#data = [[i for i in x[1:] if i!=''] for x in data]
		data = [[i for i in x[0:] if i!=''] for x in data]
		self.partitions = {}
		for i in range(len(data)):
			subset = np.array([x for x in data[i] if x in self.indicators])
			if np.shape(subset)==(0,):
				continue
			self.partitions['X'+str(i+1)] = subset
		return self.partitions

	def set_partitions(self):
		for sm in self.partitions.keys():
			models = self.partitions[sm]
			indic_inds = [self.indicators.index(m) for m in models]
			self.data['X'][sm] = self.data['X']['all'][:,indic_inds]

	def set_rand_splits(self, inds):
		self.partitions = {}
		for i in range(len(inds)):
			# Note: random partitions generated knowing final len of indicators
				# so don't need to check if indices match.
			self.partitions['X'+str(i+1)] = np.array([self.indicators[x] for x in inds[i]])
			self.data['X']['X'+str(i+1)] = self.data['X']['all'][:,inds[i]]
		return































