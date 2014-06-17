import os,time,logging,collections,math,itertools
import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
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
		mapping = { v:i for i,v in enumerate(uniques) }
		data[c] = data[c].map(mapping)
		dimensions[c] = len(mapping)
	return data,dimensions
def feature_003dt(feature,target_columns=['is_exciting'],atleast=10):
	columns = [
		'school_state','school_metro','school_county',
		'teacher_prefix','primary_focus_subject','primary_focus_area',
		'secondary_focus_subject','secondary_focus_area',
		'resource_type','poverty_level','grade_level',
 	]
	return sparse_encoder_004(feature,columns,atleast,target_columns)
def feature_003dtw(feature,target_columns=['is_exciting'],atleast=10,latest=10):
	columns = [
		'school_state','school_metro','school_county',
		'teacher_prefix','primary_focus_subject','primary_focus_area',
		'secondary_focus_subject','secondary_focus_area',
		'resource_type','poverty_level','grade_level',
 	]
	return sparse_encoder_004(feature,columns,atleast,target_columns,latest)

def sparse_encoder(feature,columns,atleast=30,atmost=1000000):
	'''
	@return id=> sparse discrete values
	'''
	data = feature[['projectid']+columns]
	dimensions = {}
	for c in columns:
		uniques = data[c].value_counts()
		uniques = uniques[ uniques>atleast ][:atmost].index
		mapping = { v:i+1 for i,v in enumerate(uniques) }
		data[c] = data[c].map(mapping).fillna(0)
		dimensions[c] = len(mapping)+1
	return data,dimensions
def sparse_encoder_002(feature,columns,dim,step):
	'''
	@return id => sparsed continuous values in projects.csv
	'''
 	data = feature[['projectid']+columns]
	dimensions = {}
	for c in columns:
		m,s = data[c].mean(),data[c].std()
		data[c] = (data[c]-m)/(s*step)+dim/2
		data[c] = data[c].map(lambda x:min(dim+1,max(0,x)))
		data[c] = np.floor(data[c])
		dimensions[c] = dim+2
	return data,dimensions
def sparse_encoder_003(feature,columns,atleast,target_columns=['is_exciting']):
	'''
	@return id=> pos% in WHOLE data of discrete values
	'''
	data = feature[['projectid']+columns]
	outcomes = pd.read_csv(os.path.join(utils.DATA_DIR,'outcomes.csv'),true_values='t',false_values='f').fillna(0)
	for c in outcomes.columns:
		if c != 'projectid':
			outcomes[c] = map(int,outcomes[c])
	data = pd.merge(data,outcomes,how='left').fillna(0)
	dimensions = {}
	for c in columns:
		# filter values
		uniques = data[c].value_counts()
		uniques = uniques[ uniques>atleast ].index
		mapping = { v:i+1 for i,v in enumerate(uniques) }
		data[c] = data[c].map(mapping).fillna(0)
		# state positive rate
		rate = data[[c]+target_columns].groupby(c).mean()
		rate.columns = [ '%s@%s'%(t,c) for t in target_columns ]
		dimensions.update({'%s@%s'%(t,c):1 for t in target_columns})
		rate[c] = rate.index
		data = pd.merge(data,rate)
	return data[['projectid']+dimensions.keys()],dimensions
def sparse_encoder_004(feature,columns,atleast=0,target_columns=['is_exciting'],latest=None):
	'''
	@return id=> pos%,cnt in the PAST of discrete values,date
	'''
	data = feature[['projectid','date_posted']+columns]
	logging.info('sparse_encoder_004 begining, data shape=%s'%(data.shape,))
	outcomes = pd.read_csv(os.path.join(utils.DATA_DIR,'outcomes.csv'),true_values='t',false_values='f').fillna(0)
	for c in outcomes.columns:
		if c != 'projectid':
			outcomes[c] = map(int,outcomes[c])
	data = pd.merge(data,outcomes,how='left').fillna(0)
	logging.info('sparse_encoder_004 outcomes joined, data shape=%s'%(data.shape,))
	dimensions = {}
	for c in columns:
		# filter values
		uniques = data[c].value_counts()
		uniques = uniques[ uniques>atleast ].index
		mapping = { v:i+1 for i,v in enumerate(uniques) }
		data[c] = data[c].map(mapping).fillna(0)
		# state positive rate,cnts
		part = data[[c,'date_posted']+target_columns]

		if latest ==None :
			cnt = 'cnt@%s'%(c,)
			def transform(x):
				x = x.groupby('date_posted').sum().reset_index()[['date_posted']+target_columns]
				x = x.sort('date_posted')
				x[cnt] = range(x.shape[0])
				x[target_columns] = (x[target_columns].cumsum().div(x[cnt]+1,axis='index')).shift(1).fillna(0)
				return x[target_columns+[cnt,'date_posted']]
		else :
			def transform(x):
				x = x.groupby('date_posted').sum().reset_index()[['date_posted']+target_columns]
				x = x.sort('date_posted')
				x[target_columns] = pd.rolling_mean(x[target_columns],latest,min_periods=0).shift(1).fillna(0)
				return x[target_columns+['date_posted']]
		stats = part.groupby(c).apply(transform)
		stats = stats.reset_index().rename(columns={ t:'%s@%s.se004.%s'%(t,c,latest) for t in target_columns })
		name_columns = [ '%s@%s.se004.%s'%(t,c,latest) for t in target_columns ]
		if latest == None:
			name_columns = name_columns + ['cnt@%s'%(c,)]
		dimensions.update({ t:1 for t in name_columns })
		logging.info('%s shape of stats=%s'%(c,stats.shape,))
		
		stats = stats[[c,'date_posted']+name_columns]
		data  = pd.merge(data,stats,how='left',on=[c,'date_posted']).fillna(0)
		logging.info('sparse_encoder_004 stats on %s ready, data shape=%s'%(c,data.shape))
	return data[['projectid']+dimensions.keys()],dimensions

def feature_004(feature,dim=40,step=0.2):
	columns = [
		'school_latitude','school_longitude',
		'total_price_excluding_optional_support','total_price_including_optional_support',
		'students_reached',
 	]
	return sparse_encoder_002(feature,columns,dim,step)
def feature_004a(feature,dim=40,step=0.2):
	columns = [
		'school_latitude','school_longitude',
		'total_price_excluding_optional_support','total_price_including_optional_support',
		'students_reached','fulfillment_labor_materials'
 	]
	return sparse_encoder_002(feature,columns,dim,step)
def feature_004d(feature):
	columns = [
		'school_latitude','school_longitude',
		'total_price_excluding_optional_support','total_price_including_optional_support',
		'students_reached','fulfillment_labor_materials'
 	]
	return feature[['projectid']+columns],{ c:1 for c in columns }

def feature_005(feature,atleast=100):
	columns = ['teacher_acctid','schoolid','school_city','school_district']
	return sparse_encoder(feature,columns,atleast)
def feature_006(feature):
	'''
	@return id => date posted time feature
	'''
	dat = feature.date_posted
	data = pd.DataFrame( feature['projectid'] )
	data['month'] 		= dat.map(lambda x:int(x[5:7]))
	data['day']			= dat.map(lambda x:int(x[8:10]))
	data['season']  	= data['month']/4
	data['abs_season']	= data['season'] + dat.map(lambda x:int(x[2:4]))*4
	dimensions = { 'month':13,'day':32,'season':5,'abs_season':1 }
	return data,dimensions
def feature_006d(feature):
	dat = feature.date_posted
	data = pd.DataFrame( feature['projectid'] )
	data['month'] 		= dat.map(lambda x:int(x[5:7]))
	data['day']			= dat.map(lambda x:int(x[8:10]))
	data['season']  	= data['month']/4
	data['abs_season']	= data['season'] + dat.map(lambda x:int(x[2:4]))*4
	dimensions = { 'month':1,'day':1,'season':1,'abs_season':1 }
	return data,dimensions

def feature_007(feature,atleast=10):
	columns = [
			'teacher_acctid',
			'schoolid','school_city','school_state','school_metro','school_district','school_county',
			'teacher_prefix',
			'primary_focus_subject','primary_focus_area',
			'secondary_focus_subject','secondary_focus_area',
			'resource_type',
			'poverty_level','grade_level',
	]
	return sparse_encoder(feature,columns,atleast)
def feature_007d(feature,atleast=10,target_columns=['is_exciting']):
	''' @return id=> pos% of discrete features'''
	columns = [
			'teacher_acctid',
			'schoolid','school_city','school_state','school_metro','school_district','school_county',
			'teacher_prefix',
			'primary_focus_subject','primary_focus_area',
			'secondary_focus_subject','secondary_focus_area',
			'resource_type',
			'poverty_level','grade_level',
	]
	return sparse_encoder_003(feature,columns,atleast,target_columns)
def feature_007dt(feature,target_columns=['is_exciting'],atleast=0):
	''' @return id=> pos% of discrete features'''
	columns = [
			'teacher_acctid',
			'schoolid','school_city','school_state','school_metro','school_district','school_county',
			'teacher_prefix',
			'primary_focus_subject','primary_focus_area',
			'secondary_focus_subject','secondary_focus_area',
			'resource_type',
			'poverty_level','grade_level',
		]
	return sparse_encoder_004(feature,columns,atleast,target_columns)
def feature_007dtw(feature,target_columns=['is_exciting'],atleast=0,latest=None):
	''' @return id=> pos% of discrete features'''
	columns = [
			'teacher_acctid',
			'schoolid','school_city','school_state','school_metro','school_district','school_county',
			'teacher_prefix',
			'primary_focus_subject','primary_focus_area',
			'secondary_focus_subject','secondary_focus_area',
			'resource_type',
			'poverty_level','grade_level',
		]
	return sparse_encoder_004(feature,columns,atleast,target_columns,latest)

def feature_008(feature,dim=40,step=0.2):
	''' @return id => sparse encoded text lengths in essays.csv '''
	essays = utils.read_csv('essays.csv')   #pd.read_csv(os.path.join(utils.DATA_DIR,'essays.csv'))
	data = pd.DataFrame(feature.projectid)
	columns = ['title','short_description','need_statement','essay']
	for c in columns :
		data['length_%s'%(c)] = essays[c].fillna('').map(len)
	columns = [ 'length_%s'%(c) for c in columns ]
	return sparse_encoder_002(data,columns,dim,step)
def tfidf_encoder(filename,columns,max_df,min_df,max_features):
	df = utils.read_csv('essays.csv')
	data = pd.DataFrame(df.projectid)
	dimemsions = {}
	for c in columns :
		texts = df[c].fillna('')
		model  = TfidfVectorizer(max_df=max_df,min_df=min_df,max_features=max_features)
		model.fit(texts)
		vector = model.transform(texts)
		c_name = '%s:tfidf_%s'%(filename,c)
		data[c_name] = map(lambda x:{ k:x[0,k] for k in x.nonzero()[1] },vector)
		dimemsions[c_name] = vector.shape[1]
	return data,dimemsions

def feature_009(feature,max_df=0.5,min_df=100,max_features=5000):
	''' @return id => title vector of essays.csv '''
	return tfidf_encoder('essay.csv',['title','short_description','need_statement'],max_df,min_df,max_features)

def nonlinear_001(x):
	return x**2
def nonlinear_002(x):
	return x**3
def nonlinear_003(x):
	return np.exp(x)
def nonlinear_004(x):
	return np.log(x-x.min()+1)

def collect_features(feature,versions=[],IDNames=[]):
	X = feature[IDNames]
	columns = {}
	for args in versions:
		v = args[0]
		f = globals()['feature_%s'%(v)](feature,*args[1:])
		if type(f) == tuple :
			columns.update(f[1])
			f = f[0]
		X = pd.merge(X,f,on=IDNames)
	X.fillna(0,inplace=True)
	return X,columns

@decorators.disk_cached(utils.CACHE_DIR+'/dense_features')
def make_dense_instance(versions=[]):
	'''
	@return train_X,train_Y,train_ID,test_X,test_ID 
	'''
	feature = pd.read_csv(os.path.join(utils.DATA_DIR,'projects.csv'),true_values='t',false_values='f')
	IDNames = ['projectid']
	ID_DATE = feature[['projectid','date_posted']]

	X,columns = collect_features(feature,versions,IDNames)

	X_Date = pd.merge(X,ID_DATE)
	outcomes = pd.read_csv(os.path.join(utils.DATA_DIR,'outcomes.csv'))
	X_Date = pd.merge(X_Date,outcomes,how='left')
	X_Date = X_Date.sort(['date_posted','projectid'])
	X_Date.index = range(X_Date.shape[0])
	
	X = np.array(X_Date[columns.keys()])
	# spilt train/test set
	train_X 	= X[  [ i  for i,t in enumerate(X_Date['date_posted'] <  '2014-01-01') if t ] ]
	train_Y 	= (X_Date[ X_Date['date_posted'] <  '2014-01-01' ]['is_exciting']=='t' ) + 0
	train_Y.index = range(len(train_Y))
	test_X 		= X[ [ i  for i,t in enumerate(X_Date['date_posted'] >=  '2014-01-01') if t ]  ]
	train_ID 	= X_Date[ X_Date['date_posted'] <  '2014-01-01' ]['projectid']
	test_ID 	= X_Date[ X_Date['date_posted'] >= '2014-01-01' ]['projectid']

	return train_X,train_Y,train_ID,test_X,test_ID 


@decorators.disk_cached(utils.CACHE_DIR+'/sparse_features')
def make_sparse_instance(versions=[],combination=0):
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
	
	X,columns = collect_features(feature,versions,IDNames)
	
	# transform sparse features
	initColumns = [ '%s_%d'%(n,i) if c>1 else n for n,c in columns.items() for i in range(c) ]
	columnMapping = { c:i for i,c in enumerate(initColumns) }
	
	if False:
		for i in range(combination):
			old_cs = columnMapping.keys()
			new_cs = [ '%s**%s'%(k1,k2) for i1,k1 in enumerate(old_cs) for i2,k2 in enumerate(old_cs[i1+1:]) ]
			base = len(old_cs)
			columnMapping.update({ c:base+i for i,c in enumerate(new_cs) })

	featureColumns = [ c for c in X.columns if c not in IDNames ]
	logging.info('#featuers=%d'%(len(columnMapping)))
	
	X_Date = pd.merge(X,ID_DATE)
	outcomes = pd.read_csv(os.path.join(utils.DATA_DIR,'outcomes.csv'))
	X_Date = pd.merge(X_Date,outcomes,how='left')
	X_Date = X_Date.sort(['date_posted','projectid'])
	X_Date.index = range(X_Date.shape[0])

	X = X_Date[featureColumns].fillna(0)
	data,row,col = [],[],[]

	for i in range(X.shape[0]):
		ins = X.loc[i]
		items =  [ ('%s_%d'%(c,ins[c]),1) if columns[c]>1 else (c,ins[c]) for c in featureColumns if type(ins[c])!=dict ]
		items += [ ('%s_%d'%(c,k),v)  for c in featureColumns if type(ins[c])==dict for k,v in ins[c].iteritems() ]
		data += [ v for k,v in items ]
		row  += [i]*len(items)
		col  += [ columnMapping[k] for k,v in items ]
		
		for c in range(combination):
			for i1,it1 in enumerate(items):
				k1,v1 = it1
				for i2,it2 in enumerate(items[i1+1:]):
					k2,v2 = it2
					data.append(v1*v2)
					row.append(i)
					c_name = '%s**%s'%(k1,k2)
					if c_name not in columnMapping:
						c_name = '%s**%s'%(k2,k1)
						if not c_name in columnMapping:
							columnMapping[c_name] = len(columnMapping)
					c_idx  = columnMapping[c_name]
					col.append(c_idx)
		
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

