import os,time,logging
import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.feature_extraction import DictVectorizer

import utils,decorators

#@decorators.disk_cached(utils.CACHE_DIR+'/basic_features')
def basic_features(base):
	base = pd.read_csv(os.path.join(utils.DATA_DIR,base))
	feature = pd.read_csv(os.path.join(utils.DATA_DIR,'features.csv'))
	ans = pd.merge(base,feature,on=['Store','Date','IsHoliday'])
	return ans

def feature_001(feature,*args):
	'''
	@param[in] feature DataFrame with column [u'Store', u'Dept', u'Date', u'Weekly_Sales', u'IsHoliday', u'Temperature', u'Fuel_Price', u'MarkDown1', u'MarkDown2', u'MarkDown3', u'MarkDown4', u'MarkDown5', u'CPI', u'Unemployment']
	@return DataFrame with column ['Store','Dept','Date','IsHoliday',Feature1,Feature2...]
	'''
	return feature[['Store','Dept','Date','IsHoliday','Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment']]

def feature_001f(feature):
	'''
	@return id => (FeatureFillNa) where nan is filled with median in Store
	'''
	basic = pd.read_csv(os.path.join(utils.DATA_DIR,'features.csv'))
	medians = basic.groupby(['Store']).median()
	stores = set(basic['Store'])
	for st in stores :
		basic[basic['Store']==st] = basic[basic['Store']==st].fillna(medians.loc[st])
	IDS = feature[['Store','Dept','Date','IsHoliday']]
	return pd.merge(IDS,basic,on=['Store','Date','IsHoliday'])

def encoding(xs,_type='1-vs-all'):
	'''
	@param[in] xs iterable of value
	@return  DataFrame
	'''
	def _1_vs_all(x):
		v = DictVectorizer()
		x = map(lambda k:{k:1},x)
		return v.fit_transform(x)
	return {
		'1-vs-all' : _1_vs_all	
	}[_type](xs)

@decorators.disk_cached(utils.CACHE_DIR+'/date_mapping_001')
def date_mapping_001(f,t):
	'''@return DataFrame['DateString','week of year','week of month','month of year'] '''
	dates = map(lambda x:x.timetuple(),utils.get_periods(f,t))
	week_of_year = map(lambda x:(x.tm_yday-1)/7,dates)
	week_of_month = map(lambda x:(x.tm_mday-1)/7,dates)
	months = map(lambda x:x.tm_mon-1,dates)
	names = map(lambda x:time.strftime('%Y-%m-%d',x),dates)
	return pd.DataFrame.from_dict({'Date':names,'WeekMonth':week_of_month,'WeekYear':week_of_year,'Month':months});

@decorators.disk_cached(utils.CACHE_DIR+'/date_mapping_002')
def date_mapping_002(f,t):
	'''@return Dataframe sparse encoding of date_mapping_001'''
	mapping = date_mapping_001(f,t)
	df = mapping[['Date']]
	for key in filter(lambda f:f!='Date',mapping):
		encoded = encoding(mapping[key])
		buff = pd.DataFrame(encoded)
		buff.columns = map(lambda i:'%s_%d'%(key,i),range(len(buff.columns)))
		df = pd.merge(df,buff,left_index=True,right_index=True)
	return df

@decorators.disk_cached(utils.CACHE_DIR+'/date_mapping_003')
def date_mapping_003(f,t):
	'''@return DataFrame['DateString','week since f','month since f','season since f','year since f']  '''
	days 	= utils.get_periods(f,t)
	weeks 	= map(lambda x:(x-days[0]).days/7,days)
	months 	= map(lambda x:(x-days[0]).days/30,days)
	seasons = map(lambda x:(x-days[0]).days/90,days)
	years	= map(lambda x:(x-days[0]).days/365,days)
	dates = map(lambda x:x.timetuple(),days)
	names = map(lambda x:time.strftime('%Y-%m-%d',x),dates)
	return pd.DataFrame.from_dict({'Date':names,'WeekA':weeks,'MonthA':months,'SeasonA':seasons,'YearA':years});

def feature_002(feature):
	'''
	@return id => (month,week)
	'''
	IDS = feature[['Store','Dept','Date','IsHoliday']]
	mapping = date_mapping_002('2010-01-01','2014-12-31')
	return pd.merge(IDS,mapping)
def feature_002s(feature):
	IDS = feature[['Store','Dept','Date','IsHoliday']]
	mapping = date_mapping_001('2010-01-01','2014-12-31')
	return pd.merge(IDS,mapping),True,{'WeekMonth':5,'WeekYear':53,'Month':12}

@decorators.disk_cached(utils.CACHE_DIR+'/id_mapping_001')
def id_mapping_001(name,max_id):
	''' 1-dimension id mapping '''
	encoded = encoding(range(max_id))
	stores = pd.DataFrame(encoded)
	stores.columns = map(lambda i:'%s_%d'%(name,i),range(len(stores.columns)))
	stores[name] = range(max_id)
	return stores

@decorators.disk_cached(utils.CACHE_DIR+'/id_mapping_002')
def id_mapping_002(names,max_ids):
	''' multi-dimension id mapping '''
	max_id = reduce((lambda a,b:a*b),max_ids,1)
	encoded = encoding(range(max_id))
	ans = pd.DataFrame(encoded)
	def make_columns(idx,cols=[]):
		if idx>=len(names): return cols
		if idx==0 : 
			return make_columns(1,[ '%s_%d'%(names[0],i) for i in xrange(max_ids[0]) ])
		ans = [ '%s-%s_%d'%(c,names[idx],i) for c in cols for i in xrange(max_ids[idx]) ]
		return make_columns(idx+1,ans)
	columns = make_columns(0)
	ans.columns = columns
	for i in range(len(names)):
		base = reduce((lambda a,b:a*b),max_ids[i+1:],1)
		rept = reduce((lambda a,b:a*b),max_ids[:i],1)
		name = names[i]
		ans[name] = [ j for r in xrange(rept) for j in xrange(max_ids[i]) for b in xrange(base) ]
	return ans	

def id_mapping_003(feature,names,max_ids):
	''' 
	multi-dimension id mapping of sparse feature return format 
	@param[in] feature DataFrame object contains ['Store','Dept','Date','IsHoliday'] and other feature columns
	@param[in] names [ [column_name_need_to_be_mapped ] ] 
	@param[in] max_ids [ [ range_of_mapping_column ] ]
	'''
	ans = feature[['Store','Dept','Date','IsHoliday']]
	dimensions = {}
	for i in range(len(names)):
		name = names[i]
		num_id = max_ids[i]

		max_id = reduce((lambda a,b:a*b),num_id,1)
		colName = '-'.join(name)
		dimensions[colName] = max_id
		ans[colName] = 1
		for i,n in enumerate(name) :
			base = reduce((lambda a,b:a*b),num_id[i+1:],1)
	return ans,True,dimensions

def feature_003(feature):
	'''
	@return id => (storeid,deptid)
	'''
	IDS = feature[['Store','Dept','Date','IsHoliday']]
	stores = id_mapping_001('Store',50)
	depts = id_mapping_001('Dept',100)
	ans = pd.merge(IDS,stores)
	ans = pd.merge(ans,depts)
	return ans
def feature_003s(feature):
	ans = feature[['Store','Dept','Date','IsHoliday']]
	ans['StoreID'] = feature['Store']
	ans['DeptID'] = feature['Dept']
	return ans,True,{'StoreID':50,'DeptID':100}

def feature_004(feature):
	'''
	@return id => (deptid_storeid)
	'''
	IDS = feature[['Store','Dept','Date','IsHoliday']]
	f = id_mapping_002(['Store','Dept'],[50,100])
	ans = pd.merge(IDS,f)
	return ans
def feature_004s(feature):
	return id_mapping_003(feature,[['Store','Dept']],[[50,100]] )

def feature_005(feature):
	'''
	@return id => (deptid_IsHoliday,storeid_IsHoliday)
	'''
	IDS = feature[['Store','Dept','Date','IsHoliday']]
	stores = id_mapping_002(['Store','IsHoliday'],[50,2])
	depts  = id_mapping_002(['Dept','IsHoliday'],[100,2])
	ans = pd.merge(IDS,stores)
	ans = pd.merge(ans,depts)
	return ans
def feature_005s(feature):
	return id_mapping_003(feature,[['Store','IsHoliday'],['Dept','IsHoliday']],[[50,2],[100,2]])
def feature_005Ss(feature):
	return id_mapping_003(feature,[['Store','IsHoliday']],[[50,2]])

def feature_006s(feature):
	'''@return id => (deptid_week,storeid_week) '''
	IDS = feature[['Store','Dept','Date','IsHoliday']]
	mapping = date_mapping_001('2010-01-01','2014-12-31')
	ans = pd.merge(IDS,mapping)   #,True,{'WeekMonth':5,'WeekYear':53,'Month':12}
	return id_mapping_003(ans,[['Store','WeekYear'],['Dept','WeekYear']],[[50,53],[100,53]])
def feature_006Ss(feature):
	'''@return id => (deptid_week,storeid_week) '''
	IDS = feature[['Store','Dept','Date','IsHoliday']]
	mapping = date_mapping_001('2010-01-01','2014-12-31')
	ans = pd.merge(IDS,mapping)   #,True,{'WeekMonth':5,'WeekYear':53,'Month':12}
	return id_mapping_003(ans,[['Store','WeekYear']],[[50,53]])

def feature_007s(feature):
	'''@return id => (dept_markdownXis0,store_markdownXis0) '''
	IDS = feature[['Store','Dept','Date','IsHoliday']]
	mappings = []
	idxs = []
	for i in range(1,6):
		key = 'MarkDown%d_Nan'%(i)
		IDS[key] = feature['MarkDown%d'%i]==np.nan
		mappings.append(['Store',key])
		idxs.append([50,2])
		mappings.append(['Dept',key])
		idxs.append([100,2])
	return id_mapping_003(IDS,mappings,idxs)

def feature_008s(feature):
	'''
	discrete continue feature with median and std, cut into 20 buckets, std/2 per bucket
	@return id => discreted 'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment' 
	'''
	keys = ['Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment']
	# compute from feature.csv
	medians = [60.71,3.51,4743.58,364.57,36.26,1176.42,2727.14,182.76,7.81]
	stds = [18.68,0.43,9262.75,8793.58,11276.46,6792.33,13086.69,39.74,1.88]
	dimensions = { '%sDiscreted'%(key):22 for key in keys }
	ans = feature[['Store','Dept','Date','IsHoliday']]
	for i,key in enumerate(keys) :
		name = '%sDiscreted'%(key)
		ans[name] = (feature[key]-medians[i])*2/stds[i]+10
		ans[name] = ans[name].map(lambda x:min(21,max(1,x)))
		ans[name][feature[key]==np.nan] = 0
	return ans,True,dimensions

def feature_009s(feature):
	'''
	@return id => (week,month,season,year) since 2010-01-01  
	'''
	IDS 	= feature[['Store','Dept','Date','IsHoliday']]
	mapping 	= date_mapping_003('2010-01-01','2014-12-31')
	return pd.merge(IDS,mapping),True,{'WeekA':55*5,'MonthA':13*5,'SeasonA':5*5,'YearA':2*5}

def make_instance(base,versions=[]):
	'''
	@param[in] base "source file name in DATA_DIR"
	@param[in] versions [ feature versions ]
	@return X,Y,W,ID,IDString
	'''
	feature = basic_features(base)
	ans = feature[['Weekly_Sales','Store','Dept','Date','IsHoliday']] if 'Weekly_Sales' in feature else feature[['Store','Dept','Date','IsHoliday']]
	for args in versions :
		v = args[0]
		f = globals()['feature_%s'%(v)](feature,*args[1:])
		ans = pd.merge(ans,f,on=['Store','Dept','Date','IsHoliday'])
	xkey = ans.columns[4:] if 'Weekly_Sales' in ans else ans.columns[3:]
	X = ans[xkey].astype(float).fillna(0)
	Y = np.array(ans['Weekly_Sales']) if 'Weekly_Sales' in ans else np.zeros(len(X))
	W = np.array(ans['IsHoliday']*4 + 1)
	ID = ans[['Store','Dept','Date']]
	IDString = map(lambda x:'_'.join(map(str,x)),np.array(ID))
	return X,Y,W,ID,IDString

@decorators.disk_cached(utils.CACHE_DIR+'/sparse_features')
def make_sparse_instance(base,versions=[]):
	'''
	@param[in] base "source file name in DATA_DIR"
	@param[in] versions [ (feature_versions,feature_args) ]
		the specific feature factory is func(*args) => (featureDataFrame,sparse,dimensions)
		dimensions => { columnName : columnDimension  }
	@return X,Y,W,ID,IDString
	'''
	eps = 1e-10
	feature = basic_features(base)
	IDNames = ['Store','Dept','Date','IsHoliday']
	if 'Weekly_Sales' in feature : IDNames.append('Weekly_Sales')
	
	# collect X

	X = feature[IDNames]
	sparseColumns = {}
	for args in versions:
		v = args[0]
		sparse = False
		f = globals()['feature_%s'%(v)](feature,*args[1:])
		if type(f) == tuple and len(f)<=3 :
			sparse = f[1]
			if sparse :
				sparseColumns.update(f[2])
			f = f[0]
		X = pd.merge(X,f,on=['Store','Dept','Date','IsHoliday'])

	# basic infos
	IDNames = ['Store','Dept','Date']
	ID = X[IDNames]
	IDString = map(lambda x:'_'.join(map(str,x)),np.array(ID))
	W = np.array(X['IsHoliday']*4 + 1)
	Y = np.array(X['Weekly_Sales']) if 'Weekly_Sales' in X else np.zeros(len(X))

	# transform sparse features
	initColumns = [ '%s_%d'%(n,i) for n,c in sparseColumns.items() for i in range(c) ]
	initColumns += [ c for c in X.columns if c not in sparseColumns and c not in IDNames and c!='Weekly_Sales' ]
	columnMapping = { c:i for i,c in enumerate(initColumns) }
	featureColumns = [ c for c in X.columns if c not in IDNames and c!='Weekly_Sales' ]
	logging.info('#featuers=%d'%(len(columnMapping)))
	X = X[featureColumns].fillna(0)
	data,row,col = [],[],[]
	for i in range(len(Y)):
		ins = X.loc[i]
		items = [ ('%s_%d'%(c,ins[c]),1) if c in sparseColumns else (c,ins[c]) for c in featureColumns ]
		data += [ v for k,v in items ]
		row  += [i]*len(items)
		col  += [ columnMapping[k]for k,v in items ]
		if i%10000==0 :
			logging.info('%d lines generated'%i)
	X = scipy.sparse.csr_matrix((data,(row,col)))
	return X,Y,W,ID,IDString

def make_svd_feature_input(outprefix,base,versions=[]):
	'''
	@param[in] base "source file name in DATA_DIR"
	@param[in] versions [ feature versions ]
	@return 
	'''
	eps = 1e-10
	X,Y,W,ID,IDString = make_instance(base,versions)
	@decorators.write_to_file('%s.input'%(outprefix),' ')
	def input_generator():
		for i in range(len(X)):
			line = X.loc[i]
			gls  = [ '%s:%s'%(k,v) for k,v in enumerate(line) if abs(v)>eps ]
			sid  = ID['Store'][i]
			did  = ID['Dept'][i]
			s    = Y[i]/100000
			f = map(str,[s,len(gls),1,1]+gls+['%s:1'%(sid),'%s:1'%(did)])
			for k in range(W[i]):
				yield f
	input_generator()
	@decorators.write_to_file('%s.ids'%(outprefix),' ')
	def id_generator():
		for i,line in enumerate(IDString):
			for k in range(W[i]):
				yield line
	id_generator()

def run_make_svd():
	make_svd_feature_input('../temp/svdfeature','train.csv',[('001',),('002',)])
	make_svd_feature_input('../temp/svdfeature_test','test.csv',[('001',),('002',)])


if __name__ == '__main__':
	run_make_svd()

