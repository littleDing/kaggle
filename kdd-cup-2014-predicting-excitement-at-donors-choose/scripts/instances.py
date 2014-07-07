import os,time,logging,collections,math,itertools,datetime
import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingRegressor
import sklearn.metrics
from sklearn.cross_validation import KFold
from sklearn import linear_model

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
	return sparse_encoder_004('projects.csv',columns,atleast,target_columns)
def feature_003dtw(feature,target_columns=['is_exciting'],atleast=10,latest=10):
	columns = [
		'school_state','school_metro','school_county',
		'teacher_prefix','primary_focus_subject','primary_focus_area',
		'secondary_focus_subject','secondary_focus_area',
		'resource_type','poverty_level','grade_level',
 	]
	return sparse_encoder_004('projects.csv',columns,atleast,target_columns,latest)

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

def prepare_data(filename,columns,target_columns=['is_exciting'],fillna=0):
	feature = utils.read_csv(filename)
	if filename != 'projects.csv' :
		pids = utils.read_csv('projects.csv')[['projectid','date_posted']]
		feature = pd.merge(pids,feature,how='left')
	data = feature[['projectid','date_posted']+columns]
	logging.info('prepare_data begining, data shape=%s'%(data.shape,))
	outcomes = utils.read_csv('outcomes.csv')
	outcomes = outcomes[['projectid']+target_columns] 
	for c in outcomes.columns:
		if c != 'projectid':
			outcomes[c] = map(int,outcomes[c].fillna(0))
	data = pd.merge(data,outcomes,how='left')
	if fillna != None:
		data = data.fillna(fillna)
	return data

def filter_values(data,c,atleast):
	uniques = data[c].value_counts()
	uniques = uniques[ uniques>atleast ].index
	mapping = { v:i+1 for i,v in enumerate(uniques) }
	data[c] = data[c].map(mapping).fillna(0)

def transformer(c,target_columns,latest,fillna):
	if latest != None:
		if fillna == None:
			def transform(x):
				x = x.groupby('date_posted').sum().reset_index()[['date_posted']+target_columns]
				x = x.sort('date_posted')
				x[target_columns] = pd.rolling_mean(x[target_columns],latest,min_periods=0).shift(1)
				x = x.fillna(0)
				return x[target_columns+['date_posted']]
		else :
			def transform(x):
				x = x.groupby('date_posted').sum().reset_index()[['date_posted']+target_columns]
				x = x.sort('date_posted')
				x[target_columns] = pd.rolling_mean(x[target_columns],latest,min_periods=0).shift(1)
				x = x.fillna(method=fillna).fillna(0)
				return x[target_columns+['date_posted']]
	else :
		cnt = 'cnt@%s'%(c,)
		if fillna == None:
			def transform(x):
				x = x.groupby('date_posted').sum().reset_index()[['date_posted']+target_columns]
				x = x.sort('date_posted')
				x[cnt] = range(x.shape[0])
				x[target_columns] = (x[target_columns].cumsum().div(x[cnt]+1,axis='index')).shift(1)
				x = x.fillna(0)
				return x[target_columns+[cnt,'date_posted']]
		else :
			def transform(x):
				x = x.groupby('date_posted').sum().reset_index()[['date_posted']+target_columns]
				x = x.sort('date_posted')
				x[cnt] = range(x.shape[0])
				x[target_columns] = (x[target_columns].cumsum().div(x[cnt]+1,axis='index')).shift(1)
				nans = x[target_columns[0]].isnull()
				x.loc[nans,cnt] = x[target_columns[0]][nans]
				x = x.fillna(method=fillna).fillna(0)
				return x[target_columns+[cnt,'date_posted']]

	return transform

def handle_column(data,c,target_columns,atleast,latest,tag,fillna=None):
	'''
	@return v,date => pos%
	'''
	filter_values(data,c,atleast)
	part = data[[c,'date_posted']+target_columns]
	# state positive rate,cnts
	transform = transformer(c,target_columns,latest,fillna)

	stats = part.groupby(c).apply(transform)
	stats = stats.reset_index().rename(columns={ t:'%s@%s.%s.%s'%(t,c,tag,latest) for t in target_columns })
	name_columns = [ '%s@%s.%s.%s'%(t,c,tag,latest) for t in target_columns ]
	if latest == None: name_columns = name_columns + ['cnt@%s'%(c,)]
	logging.info('%s shape of stats=%s'%(c,stats.shape,))
	
	stats = stats[[c,'date_posted']+name_columns]
	data  = pd.merge(data,stats,how='left',on=[c,'date_posted']).fillna(0)
	return data,name_columns

def transformer_1(c,target_columns,latest=None,circle=None,latest_days=None):
	cnt = 'cnt@%s'%(c,)
	if latest != None :
		if type(latest) == tuple:
			shift,window = latest
			def transform(x):
				x = x[['date_posted']+target_columns].sort('date_posted')
				x[target_columns] = pd.rolling_mean(x[target_columns],window,min_periods=0)
				x = x.groupby('date_posted').mean()
				x[target_columns] = x[target_columns].shift(1+shift)
				x = x.reset_index()
				x = x.fillna(method='pad').fillna(0)
				return x[target_columns+['date_posted']]
		else:
			def transform(x):
				x = x[['date_posted']+target_columns].sort('date_posted')
				x[target_columns] = pd.rolling_mean(x[target_columns],latest,min_periods=0)
				x = x.groupby('date_posted').mean()
				x[target_columns] = x[target_columns].shift(1)
				x = x.reset_index()
				x = x.fillna(method='pad').fillna(0)
				return x[target_columns+['date_posted']]
	elif circle != None :
		pass
	elif latest_days != None:
		if type(latest_days) == tuple:
			shift,window = latest_days
		else :
			shift,window = 0,latest_days
		def transform(x):
			x[cnt] = 1
			x = x.groupby('date_posted').sum().reset_index()[['date_posted',cnt]+target_columns]
			x['future'] = pd.to_datetime(x.date_posted)+datetime.timedelta(windows)
			for i in xrange(x.shape[0]):
				x[target_columns] = None
	else :
		def transform(x):
			x[cnt] = 1
			x = x.groupby('date_posted').sum().reset_index()[['date_posted',cnt]+target_columns]
			x = x.sort('date_posted')
			x[cnt] = x[cnt].cumsum()+1
			x[target_columns] = (x[target_columns].cumsum().div(x[cnt],axis='index')).shift(1)
			nans = x[target_columns[0]].isnull()
			x.loc[nans,cnt] = x[target_columns[0]][nans]
			x = x.fillna(method='pad').fillna(0)
			return x[target_columns+[cnt,'date_posted']]
	return transform

def handle_column_1(data,c,target_columns,atleast,latest,circle,tag):
	''' @return v,date => pos% '''
	filter_values(data,c,atleast)
	part = data[[c,'date_posted']+target_columns]
	# state positive rate,cnts
	transform = transformer_1(c,target_columns,latest,circle)
	stats = part.groupby(c).apply(transform)
	stats = stats.reset_index().rename(columns={ t:'%s@%s.%s.%s'%(t,c,tag,latest) for t in target_columns })
	name_columns = [ '%s@%s.%s.%s'%(t,c,tag,latest) for t in target_columns ]
	if latest == None: name_columns = name_columns + ['cnt@%s'%(c,)]
	logging.info('%s shape of stats=%s'%(c,stats.shape,))
	stats = stats[[c,'date_posted']+name_columns]
	data  = pd.merge(data,stats,how='left',on=[c,'date_posted']).fillna(0)
	return data,name_columns

@decorators.disk_cached(utils.CACHE_DIR+'/sparse_encoder_004') 
def sparse_encoder_004(filename,columns,atleast=0,target_columns=['is_exciting'],latest=None,
		outcome_na=0,groupby_na=None):
	'''
	@return id=> pos%,cnt in the PAST of discrete values,date
	'''
	data = prepare_data(filename,columns,target_columns,outcome_na)
	logging.info('sparse_encoder_004 outcomes joined, data shape=%s'%(data.shape,))
	dimensions = {}
	for c in columns:
		data,name_columns = handle_column(data,c,target_columns,atleast,latest,'se004',groupby_na)
		dimensions.update({ t:1 for t in name_columns })
		logging.info('sparse_encoder_004 stats on %s ready, data shape=%s'%(c,data.shape))
	return data[['projectid']+dimensions.keys()],dimensions

@decorators.disk_cached(utils.CACHE_DIR+'/sparse_encoder_004_1') 
def sparse_encoder_004_1(filename,columns,atleast=0,target_columns=['is_exciting'],latest=None,circle=None):
	'''
	@return id=> pos%,cnt in the PAST of discrete values,date
	'''
	data = prepare_data(filename,columns,target_columns,None)
	logging.info('sparse_encoder_004_1 outcomes joined, data shape=%s'%(data.shape,))
	dimensions = {}
	for c in columns:
		data,name_columns = handle_column_1(data,c,target_columns,atleast,latest,circle,'se004_1')
		dimensions.update({ t:1 for t in name_columns })
		logging.info('sparse_encoder_004_1 stats on %s ready, data shape=%s'%(c,data.shape))
	return data[['projectid']+dimensions.keys()],dimensions

@decorators.disk_cached(utils.CACHE_DIR+'/sparse_encoder_005') 
def sparse_encoder_005(filename,columns,atleast=0,target_columns=['is_exciting'],latest=None,
		outcome_na=0,groupby_na=None
		):
	'''
	@return id=> pos_mean,pos_max,cnt in the PAST of discrete SET values,date
	'''
	data = prepare_data(filename,columns,target_columns,outcome_na)
	logging.info('sparse_encoder_005 outcomes joined, data shape=%s'%(data.shape,))
	dimensions = {}
	for c in columns:
		data,name_columns = handle_column(data,c,target_columns,atleast,latest,'se005',groupby_na)
		dimensions.update({ t:1 for t in name_columns })
		logging.info('sparse_encoder_005 stats on %s ready, data shape=%s'%(c,data.shape))
	data 	= data[['projectid']+dimensions.keys()]
	means 	= data.groupby('projectid').mean().reset_index()
	maxs	= data.groupby('projectid').max().reset_index()
	data = pd.merge(means,maxs,suffixes=('.mean','.max'),on='projectid')
	dimensions = { c:1 for c in data.columns if c !='projectid' }
	logging.info('sparse_encoder_005 all data collected, data shape=%s'%(data.shape,))
	return data[['projectid']+dimensions.keys()],dimensions

@decorators.disk_cached(utils.CACHE_DIR+'/sparse_encoder_006') 
def sparse_encoder_006(filename,columns,atleast=0,recents=[],recent_days=[]):
	'''
	@return id=> active_cnt_all,[active_cnt_recents],[active_cnt_recent_days] of discrete values,date
	'''
	data = prepare_data(filename,columns,[],None)
	logging.info('sparse_encoder_006 outcomes joined, data shape=%s'%(data.shape,))
	dimensions = {}
	for c in columns:
		filter_values(data,c,atleast)
		part = data[[c,'date_posted']]
		# state positive rate,cnts
		cnt = 'cnt_all@%s'%c
		def transform(x):
			x[cnt] = 1
			x = x[[cnt,'date_posted']]
			ans = x.groupby('date_posted').sum().reset_index()
			for w in recents:
				ans['cnt_r%d@%s'%(w,c)] = pd.rolling_mean(ans[cnt],w,0)
			for w in recent_days :
				future = pd.to_datetime(ans.date_posted) + datetime.timedelta(w)
				ans['cnt_rd%d@%s'%(w,c)] = [ ans[:i][future[:i]>=d][cnt].sum() for i,d in enumerate(ans.date_posted)  ]
			ans[cnt] = ans[cnt].cumsum()
			return ans
		stats = part.groupby(c).apply(transform)
		name_columns = [ cc for cc in stats.columns if cc !='level_1' and cc != 'date_posted' ]
		stats = stats.reset_index()
		stats = stats[[c,'date_posted']+name_columns]
		data  = pd.merge(data,stats,how='left',on=[c,'date_posted']).fillna(0)

		dimensions.update({ t:1 for t in name_columns })
		logging.info('sparse_encoder_006 stats on %s ready, data shape=%s'%(c,data.shape))
	return data[['projectid']+dimensions.keys()],dimensions

@decorators.disk_cached(utils.CACHE_DIR+'/sparse_encoder_007') 
def sparse_encoder_007(filename,columns,target_columns,atleast=0,latests=None):
	'''
	@return id=> positive_rate of discrete values,date
	'''
	data = prepare_data(filename,columns,target_columns,None)
	logging.info('sparse_encoder_007 outcomes joined, data shape=%s'%(data.shape,))
	dimensions = {}
	shift,window = latests
	cnt,tc = 'cnt_all',target_columns
	for c in columns:
		filter_values(data,c,atleast)
		part = data[[c,'date_posted']+tc]
		# state positive rate,cnts
		def transform(x):
			x[cnt] = 1
			x = x.groupby('date_posted').sum().reset_index()
			x = x.sort('date_posted')
			future = pd.to_datetime(x.date_posted) + datetime.timedelta(window)
			x[tc+[cnt]] = pd.DataFrame([ x[:i][future[:i]>=d][tc+[cnt]].sum() for i,d in enumerate(future) ])
			x[tc] = x[tc].div(x[cnt],axis='index')
			x[tc+[cnt]] = x[tc+[cnt]].shift(window).fillna(method='pad').fillna(0)
			return x[tc+[cnt,'date_posted']]
		stats = part.groupby(c).apply(transform)
		name_columns = [ cc for cc in stats.columns if cc !='level_1' and cc != 'date_posted' ]
		stats = stats.rename(columns={ cc:'%s@%s.s%s.w%s.se007'%(cc,c,shift,window)  for cc in name_columns })
		name_columns = [ '%s@%s.s%s.w%s.se007'%(cc,c,shift,window)  for cc in name_columns ]
		stats = stats.reset_index()
		logging.info(stats.columns)
		stats = stats[[c,'date_posted']+name_columns]
		data  = pd.merge(data,stats,how='left',on=[c,'date_posted']).fillna(0)

		dimensions.update({ t:1 for t in name_columns })
		logging.info('sparse_encoder_007 stats on %s ready, data shape=%s'%(c,data.shape))
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
def feature_004d_1(feature):
	mapping = {
			'poverty_level' : {'highest poverty':1, 'high poverty':2, 'moderate poverty':3, 'low poverty':4 },
			'grade_level' : { 'Grades PreK-2':1, 'Grades 3-5':2, 'Grades 6-8':3, 'Grades 9-12':4   },
	}
	data = pd.DataFrame(feature['projectid'])
	dimensions = {}
	for c in mapping :
		after = '%s@004d_1'%(c)
		data[after] = feature[c].map(mapping[c]).fillna(-1)
		dimensions[after] = 1
	return data,dimensions

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

project_id_columns = [
			'teacher_acctid',
			'schoolid','school_city','school_state','school_metro','school_district','school_county',
			'teacher_prefix',
			'primary_focus_subject','primary_focus_area',
			'secondary_focus_subject','secondary_focus_area',
			'resource_type',
			'poverty_level','grade_level',
	]
project_id_columns_small_first = [
			'resource_type','teacher_prefix',
			'teacher_acctid',
			'schoolid','school_city','school_state','school_metro','school_district','school_county',
			'primary_focus_subject','primary_focus_area',
			'secondary_focus_subject','secondary_focus_area',
			'poverty_level','grade_level',
	]
def feature_007(feature,atleast=10):
	return sparse_encoder(feature,project_id_columns,atleast)
def feature_007d(feature,atleast=10,target_columns=['is_exciting']):
	''' @return id=> pos% of discrete features'''
	return sparse_encoder_003(feature,project_id_columns,atleast,target_columns)
def feature_007dt(feature,target_columns=['is_exciting'],atleast=0,include_all=False):
	''' @return id=> pos% of discrete features'''
	columns = project_id_columns
	if include_all :
		columns = columns + ['school_zip','school_ncesid']
	return sparse_encoder_004('projects.csv',columns,atleast,target_columns)
def feature_007dtna(feature,target_columns=['is_exciting'],atleast=0,fillna='pad',include_all=True):
	''' @return id=> pos% of discrete features'''
	columns = project_id_columns_small_first
	if include_all :
		columns = columns + ['school_zip','school_ncesid']
	return sparse_encoder_004('projects.csv',columns,atleast,target_columns,None,None,fillna)
def feature_007dtna_1(feature,target_columns=['is_exciting'],atleast=0,include_all=False):
	''' @return id=> pos% of discrete features'''
	columns = project_id_columns_small_first
	if include_all :
		columns = columns + ['school_zip','school_ncesid']
	return sparse_encoder_004_1('projects.csv',columns,atleast,target_columns)

def feature_007dtw(feature,target_columns=['is_exciting'],atleast=0,latest=None,include_all=True):
	''' @return id=> pos% of discrete features'''
	columns = project_id_columns
	if include_all :
		columns += ['school_zip','school_ncesid']
	return sparse_encoder_004('projects.csv',columns,atleast,target_columns,latest)
def feature_007dtwna(feature,target_columns=['is_exciting'],atleast=0,latest=None,fillna='pad',include_all=True):
	''' @return id=> pos% of discrete features'''
	columns = project_id_columns
	if include_all :
		columns += ['school_zip','school_ncesid']
	return sparse_encoder_004('projects.csv',columns,atleast,target_columns,latest,None,fillna)
def feature_007dtwna_1(feature,target_columns=['is_exciting'],atleast=0,latest=None,include_all=False):
	''' @return id=> pos% of discrete features'''
	columns = project_id_columns
	if include_all : columns += ['school_zip','school_ncesid']
	return sparse_encoder_004_1('projects.csv',columns,atleast,target_columns,latest)
def feature_007dtw_2(feature,target_columns=['is_exciting'],atleast=0,latest=(1,10)):
	columns = project_id_columns_small_first
	return sparse_encoder_007('projects.csv',columns,target_columns,atleast,latest)

def feature_017(feature,recents=[],recent_days=[],atleast=0):
	''' @return id=> active counts '''
	columns = project_id_columns_small_first
	return sparse_encoder_006('projects.csv',columns,atleast,recents,recent_days)
def feature_017s(feature,recents=[],recent_days=[],atleast=0):
	''' @return id=> active counts '''
	columns = ['resource_type','teacher_prefix']
	return sparse_encoder_006('projects.csv',columns,atleast,recents,recent_days)


def feature_008(feature,dim=40,step=0.2):
	''' @return id => sparse encoded text lengths in essays.csv '''
	essays = utils.read_csv('essays.csv')   #pd.read_csv(os.path.join(utils.DATA_DIR,'essays.csv'))
	data = pd.DataFrame(feature.projectid)
	columns = ['title','short_description','need_statement','essay']
	for c in columns :
		data['length_%s'%(c)] = essays[c].fillna('').map(len)
	columns = [ 'length_%s'%(c) for c in columns ]
	return sparse_encoder_002(data,columns,dim,step)
def feature_008d(feature):
	''' @return id => sparse encoded text lengths in essays.csv '''
	essays = utils.read_csv('essays.csv')   #pd.read_csv(os.path.join(utils.DATA_DIR,'essays.csv'))
	data = pd.DataFrame(feature.projectid)
	columns = ['title','short_description','need_statement','essay']
	for c in columns :
		data['length_%s'%(c)] = essays[c].fillna('').map(len)
	dimensions = { 'length_%s'%(c):1 for c in columns }
	return data,dimensions

def tfidf_encoder(filename,columns,max_df,min_df,max_features):
	df = utils.read_csv('essays.csv')
	data = pd.DataFrame(df.projectid)
	dimensions = {}
	for c in columns :
		texts = df[c].fillna('')
		model  = TfidfVectorizer(max_df=max_df,min_df=min_df,max_features=max_features)
		model.fit(texts)
		vector = model.transform(texts)
		c_name = '%s:tfidf_%s'%(filename,c)
		data[c_name] = map(lambda x:{ k:x[0,k] for k in x.nonzero()[1] },vector)
		dimensions[c_name] = vector.shape[1]
	return data,dimensions
def tfidf_encoder_001(filename,columns,max_df,min_df,max_features):
	'''
		make tfidf vectors and state for gbdt
		return id => { num_words,sum_tfidf } for every column
	'''
	df = utils.read_csv(filename)
	data = pd.DataFrame(df.projectid)
	dimensions = {}
	for c in columns :
		texts = df[c].fillna('')
		model  = TfidfVectorizer(max_df=max_df,min_df=min_df,max_features=max_features)
		model.fit(texts)
		vector = model.transform(texts)
		stats = [ '%s@%s'%(s,c) for s in ['#words','sum_tfidf'] ]
		data[stats[0]] = [ vector[i].nonzero()[0].shape[0] for i in range(vector.shape[0]) ]
		data[stats[1]] = vector.sum(1)
		dimensions.update({ s:1 for s in stats })
	return data,dimensions

@decorators.disk_cached(utils.CACHE_DIR+'/tfidf_encoder_002')
def tfidf_encoder_002(filename,columns,target_columns,max_df,min_df,max_features):
	'''
		state positive rate on words and sum with tfidf weight
		return id => { mean(pos_rate),sum(tfidf*pos_rate)/sum(tfidf) }
	'''
	df = prepare_data(filename,columns,target_columns,None)
	data = pd.DataFrame(df.projectid)
	dimensions = {}
	for c in columns :
		texts = df[c].fillna('')
		model = TfidfVectorizer(max_df=max_df,min_df=min_df,max_features=max_features)
		model.fit(texts)
		vector = model.transform(texts)
		data['tfidf@%s'%c] = [ { c:v[0,c] for c in v.nonzero()[1] } for v in vector ]
		def transformer(x):
			pass

def feature_009(feature,max_df=0.5,min_df=100,max_features=5000):
	''' @return id => title vector of essays.csv '''
	return tfidf_encoder('essays.csv',['title','short_description','need_statement'],max_df,min_df,max_features)
def feature_009d(feature,columns=['title'],max_df=0.5,min_df=10,max_features=5000):
	return tfidf_encoder_001('essays.csv',columns,max_df,min_df,max_features)

@decorators.disk_cached(utils.CACHE_DIR+'/feature_020')
def _feature_020():
	'''
		basic infos from resource.csv
		@return id => [resouceid_cnt,vendor_cnt,project_resource_type_cnt,
				item_quantity_sum,item_quantity_max,
				item_unit_price_mean,item_unit_price_max
				]
	'''
	resources = utils.read_csv('resources.csv')
	gp = resources.groupby('projectid')
	def mapper(x):
		return pd.Series([
			x.shape[0],
			x.vendorid.unique().shape[0],
			x.project_resource_type.unique().shape[0],
			x.item_quantity.sum(),x.item_quantity.max(),
			x.item_unit_price.mean(),x.item_unit_price.max(),
		])
	data = gp.apply(mapper)
	columns = [	"resouceid_cnt","vendor_cnt","project_resource_type_cnt",
				"item_quantity_sum","item_quantity_max",
			    "item_unit_price_mean","item_unit_price_max"]
	data.columns = columns
	data = data.reset_index()
	return data,columns
def feature_020(feature):
	data,columns = _feature_020()
	data = pd.merge(pd.DataFrame(feature['projectid']),data,how='left').fillna(0)
	dimensions = { c:1 for c in columns }
	return data,dimensions

def feature_021(feature,target_columns=['is_exciting'],atleast=0):
	'''
		@return id => set features in resources.csv
	'''
	columns = ['vendorid','project_resource_type']
	return sparse_encoder_005('resources.csv',columns,atleast,target_columns)

target_columns_all = [u'is_exciting', u'at_least_1_teacher_referred_donor', u'fully_funded', u'at_least_1_green_donation', u'great_chat', u'three_or_more_non_teacher_referred_donors', u'one_non_teacher_referred_donor_giving_100_plus', u'donation_from_thoughtful_donor', u'great_messages_proportion', u'teacher_referred_count', u'non_teacher_referred_count']

def feature_030_1(feature,atleast=0):
	''' @return id => pos_rate of Title '''
	columns = [u'title', u'short_description', u'need_statement', u'essay']
	return sparse_encoder_004_1('essays.csv',columns,atleast,target_columns_all)	

def feature_030_2(feature,atleast=0,recents=[],recent_days=[]):
	''' @return id => active count of Title '''
	columns = [u'title', u'short_description', u'need_statement', u'essay']
	return sparse_encoder_006('essays.csv',columns,atleast,recents,recent_days)

@decorators.disk_cached(utils.CACHE_DIR+'/tfidf_vector')
def to_tfidf_vector(field,max_df=0.5,min_df=2,max_features=5000):
	'''
	@return train_vectors,train_ids,targets,test_vectors,test_ids
	'''
	df = utils.read_csv('essays.csv')
	outcomes = utils.read_csv('outcomes.csv')[['projectid','is_exciting']]
	df = pd.merge(df,outcomes,how='left')
	texts = df[field].fillna('')
	model  = TfidfVectorizer(max_df=max_df,min_df=min_df,max_features=max_features)
	model.fit(texts)
	train_idx,test_idx = df.is_exciting.notnull(),df.is_exciting.isnull()

	train_texts = texts[train_idx]
	test_texts	= texts[test_idx]
	train_ids   = df.projectid[train_idx]
	test_ids 	= df.projectid[test_idx]
	targets		= df.is_exciting[train_idx]
	train_vector = model.transform(train_texts)
	test_vector = model.transform(test_texts)
	return train_vector,train_ids,targets,test_vector,test_ids

@decorators.disk_cached(utils.CACHE_DIR+'/lr_tfidf')
def lr_tfidf(field,max_df=0.5,min_df=2,max_features=5000):
	train_vector,train_ids,targets,test_vector,test_ids = to_tfidf_vector(field)
	model = linear_model.LogisticRegression()
	model.fit(train_vector,targets)
	train_yy = model.predict_proba(train_vector)[:,1]
	test_yy = model.predict_proba(test_vector)[:,1]
	return pd.DataFrame({'projectid':list(train_ids)+list(test_ids),'score@%s'%field:list(train_yy)+list(test_yy)})
def feature_040(feature,field):
	'''
	@return id=> prediction by logistic regression using tfidf text fields
	'''
	return lr_tfidf(field),{'score@%s'%field:1}

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
def make_dense_instance(versions=[],train_dates=None,test_dates=None):
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
	if train_dates==None: train_dates = '1900-01-01','2014-01-01'
	train_from,train_to = train_dates
	if test_dates==None: test_dates = '2014-01-01','2015-12-31'
	test_from,test_to = test_dates
	# spilt train/test set
	train_idx 	= (train_from <= X_Date['date_posted']) & ( X_Date['date_posted'] <train_to )
	test_idx 	= (test_from <= X_Date['date_posted']) & ( X_Date['date_posted'] <test_to )
	
	train_X 	= X[  [ i  for i,t in enumerate(train_idx) if t ] ]
	train_Y 	= (X_Date[ train_idx ]['is_exciting']=='t' ) + 0
	train_Y.index = range(len(train_Y))
	test_X 		= X[ [ i  for i,t in enumerate(test_idx) if t ]  ]
	train_ID 	= X_Date[ train_idx ]['projectid']
	test_ID 	= X_Date[ test_idx  ]['projectid']

	return train_X,train_Y,train_ID,test_X,test_ID 

@decorators.disk_cached(utils.CACHE_DIR+'/dense_features_raw')
def make_dense_instance_raw(versions=[]):
	'''
	@return  
	'''
	feature = pd.read_csv(os.path.join(utils.DATA_DIR,'projects.csv'),true_values='t',false_values='f')
	IDNames = ['projectid']
	ID_DATE = feature[['projectid','date_posted']]

	X,columns = collect_features(feature,versions,IDNames)

	X_Date = pd.merge(X,ID_DATE)
	outcomes = pd.read_csv(os.path.join(utils.DATA_DIR,'outcomes.csv'))
	X_Date = pd.merge(X_Date,outcomes[['projectid','is_exciting']],how='left').fillna(0)
	X_Date = X_Date.sort(['date_posted','projectid'])
	X_Date.index = range(X_Date.shape[0])
	return X_Date

def dense_instance_to_csv(versions,path):
	data = make_dense_instance_raw(versions)
	data.to_csv(path,index=False)

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

