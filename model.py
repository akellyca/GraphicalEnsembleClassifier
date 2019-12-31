import numpy as np
from operator import itemgetter
from sklearn.metrics import accuracy_score, log_loss, roc_curve
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from scipy.sparse import csr_matrix

class Model:
	def __init__(self, params):
		self.m = params['model']
		self.sms = params['sms']

		self.base_model = params['base_model']
		self.penalty = params['penalty']
		self.Cs = params['Cs']
		self.n_folds = params['n_folds']

		if self.m == 'gec':
			# Expects that GEC overlaps are notated by Xi_j / Xi_j_k
			self.coefs = { sm: (-1.)**sm.count('_') for sm in self.sms }
		else:
			self.coefs = {sm:1. for sm in self.sms}
		params['coefs'] = self.coefs

		self.params = params
		self.kf_obj = KFold(n_splits=self.n_folds, shuffle=False, random_state=0)
		return

	def train_sms(self, X, y):

		self.kf = list(self.kf_obj.split(y))
		self.trained = {}

		for sm in self.sms:
			if X[sm].shape==(1,0) or X[sm].shape==(0,):
				self.trained[sm] = []
				continue
			self.trained[sm] = {C:[] for C in self.Cs}
			for C in self.Cs: # Make fits for each CV split
				for cv in range(len(self.kf)):
					cv_x = X[sm][self.kf[cv][0], :]
					cv_y = [y[i] for i in self.kf[cv][0]]
					trained = self._fit_helper(cv_x, cv_y, C)
					self.trained[sm][C].append(trained)

		return self.trained

	def tune_sms(self, X, y, tune_type='model'):
		# given trained submodels for each C, tune as appropriate
		accs = {C:[] for C in self.Cs}
		weights = {}
		for C in self.Cs:
			for cv in range(len(self.kf)):
				# Get held-out tune set for the kth-fold
				cv_x = {sm: X[sm][self.kf[cv][1], :] if X[sm].shape!=(0,) else csr_matrix([]) for sm in self.sms}
				cv_y = [y[i] for i in self.kf[cv][1]]
				# Temporarily set fits to this C and cv for tuning training
				self.fit = {sm: self.trained[sm][C][cv] if self.trained[sm]!=[] else [] for sm in self.sms}
				cv_y_train = [y[i] for i in self.kf[cv][0]]
				#prior_odds = np.log(sum(cv_y_train)/(len(cv_y_train)-sum(cv_y_train)))
				p_y = sum(cv_y_train)/len(cv_y_train)
				pred, roc = self.predict(cv_x, p_y)
				acc = self.eval(pred, cv_y)
				accs[C].append( acc )
				self.fits, self.preds, self.rocs = {}, {}, {} #clear objects
			accs[C] = np.mean(accs[C])
		C = max(accs.items(), key=itemgetter(1))[0]
		self.C_vals = C
		self.fit = {sm:self._fit_helper(X[sm], y, C) for sm in self.sms}
		return self.fit

	def train_tune_nb(self, X, y):
		accs = {}
		self.trained = {'all':{}}
		for C in self.Cs:
			nb = BernoulliNB(alpha=C)
			cv = cross_val_score(nb, X['all'], y, cv=self.kf_obj)
			self.trained['all'][C] = cv
			accs[C] = np.mean(cv)
		nb_C = max(accs.items(), key=itemgetter(1))[0]
		fit = {'all': BernoulliNB(alpha=nb_C).fit(X['all'], y) }
		self.fit = fit
		return self.fit


	# def predict(self, X, log_odds):
	# 	self.preds, self.rocs = [], []
	# 	weights = {}
	# 	ps = []
	# 	for sm in self.sms:
	# 		if X[sm].shape==(1,0):
	# 			weights[sm] = []
	# 			continue # empty overlap set
	# 		f = self.fit[sm]
	# 		weights[sm] = [f.coef_[0], f.intercept_[0] ]

	# 	n_pts = X[list(X.keys())[0]].shape[0]
	# 	for i in range(n_pts):
	# 		W, b = [0, log_odds]
	# 		k = 0
	# 		for sm in self.sms:
	# 			if X[sm].shape==(1,0):
	# 				continue # empty overlap set
	# 			c = self.coefs[sm]		
	# 			W += c*weights[sm][0]*X[sm][i,:].T
	# 			b += c*(weights[sm][1] - log_odds)
	# 		#print( 1 / (1+np.exp(-W-b) ) )
	# 		#exit()
	# 		#exit()
	# 		ps.append( float( 1 / (1+np.exp(-W-b) )[0])  )
	# 		y = 1 if W+b>0 else 0
	# 		self.preds.append(y)
	# 		self.rocs.append( 1 / (1+np.exp(-W-b)))
	# 	#print(ps[:10])
	# 	return self.preds, self.rocs

	def predict(self, X, odds):
		self.preds, self.rocs = [], []
		n_pts = X[list(X.keys())[0]].shape[0]
		p = {}

		for sm in self.sms:
			if X[sm].shape==(1,0) or X[sm].shape==(0,):
				continue
			f = self.fit[sm]
			probs = f.predict_proba(X[sm])
			p[sm] = probs[:,1] / (probs[:,0] + probs[:,1])

		for i in range(n_pts):
			p1 = odds
			p0 = (1-odds)
			for sm in self.sms:
				if X[sm].shape==(1,0) or X[sm].shape==(0,):
					continue
				c = self.coefs[sm]
				p1 = p1*p[sm][i]/odds if c>0 else p1*odds/p[sm][i]
				p0 = p0*(1-p[sm][i])/(1-odds) if c>0 else p0*(1-odds)/(1-p[sm][i])
			prob = p1 / (p1 + p0)
			y = 1 if prob>0.5 else 0
			self.preds.append(y)
			self.rocs.append(1 - prob)
		return self.preds, self.rocs

	def eval(self, predicted, truth):
		accuracy = accuracy_score(truth, predicted) if predicted!=[] else 0.
		return accuracy

	def get_log_loss(self, predicted, truth):
		return log_loss(truth, predicted)

	def get_roc_curve(self, predicted, truth):
		return roc_curve(truth, predicted)



	### ============================================================###
		### Auxiliary functions ###
	### ============================================================###

	def _fit_helper(self, X, y, C):
		if self.base_model=='lr':
			return LogisticRegression(penalty=self.penalty, C=C).fit(X, y) if X.shape!=(0,) else []
		elif self.base_model=='svm':
			return SVC(C=C,kernel=self.penalty,degree=2,probability=True).fit(X, y) if X.shape!=(0,) else []




