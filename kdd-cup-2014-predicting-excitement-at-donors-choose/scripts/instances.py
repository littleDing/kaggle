import os,time,logging,collections,math,itertools
import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import GradientBoostingRegressor
import sklearn.metrics
from sklearn.cross_validation import KFold

import utils,decorators

@decorators.disk_cached(utils.CACHE_DIR+'/cross_validations')
def cross_validations(seed,fold):
	'''
	@return [ (train_id_set,test_id_set) ]
	'''
	ans = []
	projects = pd.read_csv(os.path.join(utils.DATA_DIR,'projects.csv'))
	all_ids  = projects[projects['date_posted'] < '2014-01-01']['projectid']
	for train_index,test_index in KFold(len(all_ids),fold,shuffle=True,random_state=seed):
		ans.append((
			set(all_ids.iloc[train_index]),
			set(all_ids.iloc[test_index]),
			))
	return ans

def feature_001(feature,*args):
	'''
	@param[in] feature DataFrame with column [u'projectid']
	@return id => 1
	'''
	projectid = feature['projectid']
	return pd.DataFrame.from_dict({'projectid':projectid,'const':1}),{'const':1}

def feature_002(feature,*args):
	'''
	@return id => sparse boolean values in projects.csv
	'''
	booleans = ['school_charter','school_magnet','school_year_round','school_nlns','school_kipp','school_charter_ready_promise',
 				'teacher_teach_for_america','teacher_ny_teaching_fellow',
 				'eligible_double_your_impact_match','eligible_almost_home_match']
	data = feature[['projectid']+booleans]
	dimensions = {}
	for c in booleans:
		data[c] = data[c] + 0
		dimensions[c] = 1
	return data,dimensions

def feature_003(feature):
	'''
	@return id => sparse discrete values in projects.csv
	'''
	columns = [
		'school_state','school_metro','school_county',
		'teacher_prefix','primary_focus_subject','primary_focus_area',
		'secondary_focus_subject','secondary_focus_area',
		'resource_type','poverty_level','grade_level',
 	]
 	data = feature[['projectid']+columns]
	dimensions = {}
	for c in columns:
		uniques = data[c].unique()
		mapping = { k:v for k,v in enumerate(uniques) }
		data[c] = data[c].map(mapping)
		dimensions[c] = len(mapping)
	return data,dimensions

def feature_004(feature,dim=40,step=0.2):
	'''
	@return id => sparsed continuous values in projects.csv
	'''
	columns = [
		'school_latitude','school_longitude',
		'total_price_excluding_optional_support','total_price_including_optional_support',
		'students_reached',
 	]
 	data = feature[['projectid']+columns]
	dimensions = {}
	for c in columns:
		m,s = data[c].mean(),data[c].std()
		data[c] = (data[c]-m)/(s*step)+dim/2
		data[c] = data[c].map(lambda x:min(dim+1,max(0,x)))
		data[c] = np.floor(data[c])
		dimensions[c] = dim+2
	return data,dimensions

def nonlinear_001(x):
	return x**2
def nonlinear_002(x):
	return x**3
def nonlinear_003(x):
	return np.exp(x)
def nonlinear_004(x):
	return np.log(x-x.min()+1)

@decorators.disk_cached(utils.CACHE_DIR+'/sparse_features')
def make_sparse_instance(versions=[]):
	'''
	@param[in] versions [ (feature_versions,feature_args) ]
		the specific feature factory is func(ids,*args) => (featureDataFrame,dimensions)
			dimensions => { columnName : columnDimension  } where columnDimension>1 will be transform to sparse column
	@return train_X,train_Y,train_ID,test_X,test_ID 
	'''
	eps = 1e-10
	feature = pd.read_csv(os.path.join(utils.DATA_DIR,'projects.csv'),true_values='t',false_values='f')
	IDNames = ['projectid']
	ID_DATE = feature[['projectid','date_posted']]

	# collect X
	X = feature[IDNames]
	columns = {}
	for args in versions:
		v = args[0]
		f = globals()['feature_%s'%(v)](feature,*args[1:])
		if type(f) == tuple :
			columns.update(f[1])
			f = f[0]
		else :
			for k in f:
				if k in IDNames : continue
				base_x = f[k]
				for func in nonlinears :
					key = '%s_nl_%s'%(k,func)
					f[key] = globals()['nonlinear_%s'%(func)](base_x)
		X = pd.merge(X,f,on=IDNames)

	# transform sparse features
	initColumns = [ '%s_%d'%(n,i) if c>1 else n for n,c in columns.items() for i in range(c) ]
	columnMapping = { c:i for i,c in enumerate(initColumns) }
	featureColumns = [ c for c in X.columns if c not in IDNames ]
	logging.info('#featuers=%d'%(len(columnMapping)))
	
	X_Date = pd.merge(X,ID_DATE)
	outcomes = pd.read_csv(os.path.join(utils.DATA_DIR,'outcomes.csv'))
	X_Date = pd.merge(X_Date,outcomes,how='left')
	X_Date = X_Date.sort('projectid')

	X = X_Date[featureColumns].fillna(0)
	data,row,col = [],[],[]
	for i in range(X.shape[0]):
		ins = X.loc[i]
		items = [ ('%s_%d'%(c,ins[c]),1) if columns[c]>1 else (c,ins[c]) for c in featureColumns ]
		data += [ v for k,v in items ]
		row  += [i]*len(items)
		col  += [ columnMapping[k] for k,v in items ]
		if i%10000==0 :
			logging.info('%d lines generated'%i)
	
	X = scipy.sparse.csr_matrix((data,(row,col)))
	
	# spilt train/test set
	train_X 	= X[  [ i  for i,t in enumerate(X_Date['date_posted'] <  '2014-01-01') if t ] ]
	train_Y 	= (X_Date[ X_Date['date_posted'] <  '2014-01-01' ]['is_exciting']=='t' ) + 0
	train_Y.index = range(len(train_Y))
	test_X 		= X[ [ i  for i,t in enumerate(X_Date['date_posted'] >=  '2014-01-01') if t ]  ]
	
	train_ID 	= X_Date[ X_Date['date_posted'] <  '2014-01-01' ]['projectid']
	test_ID 	= X_Date[ X_Date['date_posted'] >= '2014-01-01' ]['projectid']

	return train_X,train_Y,train_ID,test_X,test_ID 

if __name__ == '__main__':
	run_make_svd()

