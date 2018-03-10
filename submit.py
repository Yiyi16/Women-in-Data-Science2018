import numpy as np 
import pandas as pd
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from collections import Counter
from sklearn import svm
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, scale, normalize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from collections import Counter
#from sklearn.grid_search import GridSearchCV
import csv
import warnings
from functools import wraps

import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBClassifier
import mca

from scipy.stats import skew

#import keras.backend as K
#from keras.models import Sequential
#from keras.layers import Input, merge, Dense, Dropout, Merge
#from keras.regularizers import l2
#from keras.optimizers import SGD

def ignore_warnings(f):
    @wraps(f)
    def inner(*args, **kwargs):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore")
            response = f(*args, **kwargs)
        return response
    return inner

@ignore_warnings
def start():
	data = pd.read_csv('train.csv')
	data = data.fillna(-1)
	y = np.array(data['is_female'], np.int)
	data = data.drop(['is_female'], axis=1)

	data2 = pd.read_csv('test.csv')
	data2 = data2.fillna(-1)
	data = data.append(data2)
	print data.shape

	label_encode = LabelEncoder()
	mat = np.array([])

	for i in range(1, data.shape[1]):
		tmp = np.array(data.ix[:,i])
		tmp_label_encode = label_encode.fit_transform(tmp)
		if i == 1:
			mat = tmp_label_encode
		else:
			mat = np.vstack((mat, tmp_label_encode))
	mat_new = mat.T
	print mat_new.shape

	np.save('train.npy', mat_new[:18255,:])
	np.save('test.npy', mat_new[18255:,:])
	np.save('y.npy', y)


def GradientBoosting_Classifier():
	input = np.load('train.npy')
	label = np.load('y.npy')
	test = np.load('test.npy')

	#pca = PCA(n_components=3)
	#input = pca.fit_transform(input)
	#test = pca.fit_transform(test)

	X_train, X_test, y_train, y_test = train_test_split(input, label, test_size=0.2)
	#X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.05)
	
	grd = GradientBoostingClassifier(n_estimators=100, max_depth=1)	
	#grd_enc = OneHotEncoder()
	#grd_lm = LogisticRegression()
	grd.fit(X_train, y_train)
	#grd_enc.fit(grd.apply(X_train)[:, :, 0])
	#grd_lm.fit(grd_enc.fit(grd.apply(input)[:, :, 0]), label)
	#y_pred = grd.predict_proba(grd_enc.transform(grd.apply(test)[:, :, 0]))[:, 1]
	ypred = grd.predict_proba(X_test)[:,1]

	'''
	with open('solution.csv','w') as fout:
		fieldnames = ['test_id','is_female']
		writer = csv.DictWriter(fout, fieldnames=fieldnames)
		writer.writeheader()
		for i in range(len(y_pred_grd_lm)):
			writer.writerow({'test_id': str(i), 'is_female': y_pred_grd_lm[i]})
	'''
	
	#print y_pred_grd
	print roc_auc_score(y_test, ypred)

def my_SVM():
	input = np.load('train.npy')
	label = np.load('y.npy')
	test = np.load('test.npy')

	X_train, X_test, y_train, y_test = train_test_split(input, label, test_size=0.8)
	clf = svm.SVC()
	clf.fit(X_train, y_train)
	np.savetxt('a.txt', clf.decision_function(X_test))

	#predict = []
	#for i in range(len(X_test)):
#		predict.append(int(clf.predict_proba(X_test[i])))

#	print roc_auc_score(y_test, predict)
def lightGBM():
	input = np.load('train.npy')
	label = np.load('y.npy')
	test = np.load('test.npy')
	
	X_train, X_test, y_train, y_test = train_test_split(input, label, test_size=0.8)
	
	lgb_train = lgb.Dataset(X_train, y_train)
	params = {'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 31,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'verbose': 0}
    
	gbm = lgb.train(params, lgb_train, num_boost_round=100)

	# predict
	y_pred = gbm.predict(X_test)
	# eval
	print('The roc_auc_score of prediction is:', roc_auc_score(y_test, y_pred))

def my_xgboost():
	input = np.load('train.npy')
	label = np.load('y.npy')
	test = np.load('test.npy')



	X_train, X_test, y_train, y_test = train_test_split(input, label, test_size=0.2)
	#X_train = scale(X_train)
	#X_test = scale(X_test)

	dtrain = xgb.DMatrix(X_train, y_train)
	dtest = xgb.DMatrix(X_test)
	param = {'max_depth':100, 'eta':0.2, 'silent':1, 'objective':'binary:logistic', 'eval_metric':'auc'}
	num_round = 50
	bst = xgb.train(param, dtrain, num_round)
	ypred = bst.predict(dtest)
		



	#model = XGBClassifier(max_depth=150, learning_rate=0.1, n_estimators=200, silent=True, objective='binary:logistic', 
#			gamma=0, early_stopping_rounds=30, )
	#eval_set  = [(X_train,y_train), (X_test,y_test)]
	#n_estimators = range(50,400,50)
	#param_grid = dict(n_estimators = n_estimators)
	#grid_search = GridSearchCV(model, param_grid, scoring="auc", n_jobs = -1)
	#grid_result = grid_search.fit(K)
	#model.fit(X_train, y_train, eval_set=eval_set, eval_metric="auc")
	#ypred = model.predict_proba(X_test)[:,1]

	#dtrain = xgb.DMatrix(X_train, y_train)
	#dtest = xgb.DMatrix(X_test)
	# specify parameters via map
	#param = {'max_depth':150, 'eta':0.1, 'silent':1, 'objective':'binary:logistic', 'eval_metric':'auc'}
	#num_round = 500
	#bst = xgb.train(param, dtrain, num_round)
	# make prediction
	#ypred = bst.predict(dtest)


	print roc_auc_score(y_test, ypred)


def calculate_weight():
	train = np.load('train.npy')
	test = np.load('test.npy')
	y_train = np.zeros(train.shape[0])
	y_test = np.ones(test.shape[0])
	
	data = np.concatenate((train, test), axis=0)
	label = np.concatenate((y_train, y_test), axis=0)
	
	permutation = np.random.permutation(data.shape[0])
	print permutation
	shuffled_data = data[permutation]
	shuffled_label = label[permutation]

	
	X_train, X_test, y_train, y_test = train_test_split(shuffled_data, shuffled_label, test_size=0.9)
	clf = svm.SVR()
	clf.fit(X_train, y_train)
	ypred = clf.predict(X_test)
	weight = []
	for i in range(len(ypred)):
		weight.append(pred[i]/(1-pred[i]))
	print weight
	print roc_auc_score(y_test, ypred)
	

	'''
	kf = KFold(n_splits=2)
	weight = np.zeros(train.shape[0])
	for train_index, test_index in kf.split(shuffled_data):
		print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_test = shuffled_data[train_index], shuffled_data[test_index]
		y_train, y_test = shuffled_label[train_index], shuffled_label[test_index]
		clf = svm.SVR()
		clf.fit(X_train, y_train)
		ypred = clf.predict(X_test)
		
		for i in test_index:
			weight[i] = ypred[i]/(1-ypred[i])
		print weight
		print roc_auc_score(y_test, ypred)
	'''
	
	
def my_MLPClassifier():
	input = np.load('train.npy')
	label = np.load('y.npy')
	test = np.load('test.npy')

	X_train, X_test, y_train, y_test = train_test_split(input, label, test_size=0.2)

	mlp = MLPClassifier(hidden_layer_sizes=(input.shape[1],128,64,32), 
					activation='relu', 
     				beta_1=0.6, 
     				beta_2=0.9,
                    alpha = 0.001,
                    early_stopping = True,
                    shuffle = True,
                    warm_start = True,
                    validation_fraction = 0.3,
     				learning_rate_init=0.01, 
     				max_iter = 14000, 
     				random_state = 1235, 
     				learning_rate='adaptive')

	mlp.fit(X_train, y_train)
	print("Training set score: %f" % mlp.score(X_train, y_train))
	ypred = mlp.predict_proba(X_test)[:,1]
	print roc_auc_score(y_test, ypred)

def my_random_forest():
	input = np.load('train.npy')
	label = np.load('y.npy')
	test = np.load('test.npy')


	X_train, X_test, y_train, y_test = train_test_split(input, label, test_size=0.2)
	#X_train = scale(X_train)
	#X_test = scale(X_test)

	clf = RandomForestClassifier(n_estimators=300)
	clf = clf.fit(X_train, y_train)
	ypred = clf.predict_proba(X_test)[:,1]
	print roc_auc_score(y_test, ypred)
		

if __name__ == '__main__':
	#start()
	#GradientBoosting_Classifier()

	#lightGBM()
	#my_xgboost()
	#my_MLPClassifier()
	#my_random_forest()
	
	input = np.load('train.npy')
	label = np.load('y.npy')
	test = np.load('test.npy')


	input_new = np.array([])
	for j in range(input.shape[1]):
		col = list(input[:,j])
		unique = set(col)
		freq_dict = []
		for i in range(max(unique)+1):
			freq_dict.append(col.count(i)/float(input.shape[0]))
		#print freq_dict
		for i in range(len(col)):
			col[i] = freq_dict[col[i]]
		#print col
		input_new = np.append(input_new, col)
	input_new = np.reshape(input_new, (-1, 1234))
	np.save('input_new.npy', input_new)

	test_new = np.array([])
	for j in range(test.shape[1]):
		col = list(test[:,j])
		unique = set(col)
		freq_dict = []
		for i in range(max(unique)+1):
			freq_dict.append(col.count(i)/float(test.shape[0]))
		#print freq_dict
		for i in range(len(col)):
			col[i] = freq_dict[col[i]]
		#print col
		test_new = np.append(test_new, col)
	test_new = np.reshape(test_new, (-1, 1234))
	np.save('test_new.npy', test_new)


	





	'''
	pca = PCA(n_components=3)
	pca.fit(test)
	print pca.explained_variance_ratio_
	'''
	
	'''
	#n_samples = len(input)
	#input2 = input[:n_samples/5, :]
	#label = label[:n_samples/5]
	#print input.shape, label.shape
	drop = np.array([])
	count_train0 = 0
	for i in range(input.shape[1]):
		if skew(input[:,i]) == 0 and skew(test[:,i]) > 50:
			count_train0 += 1
			print i
			drop = np.append(drop, int(i))
			if i == 29:
				#print skew(input[:,i])
				print skew(test[:,i])
			#print skew(input[:,i])
			#print skew(test[:,i])

	count_test0 = 0
	for i in range(input.shape[1]):
		if skew(test[:,i]) == 0 and skew(input[:,i]) > 50:
			count_test0 += 1
			print i
			drop = np.append(drop,int(i))
			#print skew(input[:,i])
			#print skew(test[:,i])

	print count_test0+count_train0
	print drop
	print input.shape
	'''




		#plt.scatter(skew(input[:,i]), skew(test[:,i]))
	#plt.xlim(-150,150)
	#plt.ylim(-150,150)
	#plt.show()
		#print skew(input2[:,i])
		#print skew(input[:,i])
		#if np.absolute(skew(input2[:,i]) - skew(input[:,i])) > 0.5:
		#	print i 

	#x_emb = TSNE(n_components=2).fit_transform(input)
	#colors = ['r' if e == 0 else 'b' for e in label]
	#print colors
	#plt.scatter(x_emb[:,0], x_emb[:,1], c=colors, s=20, alpha=0.5)
	#plt.show()
	

	#print input.shape
	
	#plt.hist(test[:,2], facecolor = 'blue', alpha = 0.5, bins = 60)
	#plt.hist(input[:,2], facecolor = 'red', alpha = 0.5, bins = 60)
	#plt.show()

	'''
	count  = 0
	for i in range(1234):
		if np.absolute(np.absolute(skew(test[:,i])) - np.absolute(skew(input[:,i]))) > 2:
			print i
			count += 1
	print count
	'''
	#calculate_weight()


	


