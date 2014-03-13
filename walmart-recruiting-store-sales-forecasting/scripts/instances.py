import os,time
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer

import utils,decorators

#@decorators.disk_cached(utils.CACHE_DIR+'/basic_features')
def basic_features(base):
	base = pd.read_csv(os.path.join(utils.DATA_DIR,base))
	feature = pd.read_csv(os.path.join(utils.DATA_DIR,'features.csv'))
	return pd.merge(base,feature,on=['Store','Date','IsHoliday'])

def feature_001(feature,*args):
	'''
	@param[in] feature DataFrame with column [u'Store', u'Dept', u'Date', u'Weekly_Sales', u'IsHoliday', u'Temperature', u'Fuel_Price', u'MarkDown1', u'MarkDown2', u'MarkDown3', u'MarkDown4', u'MarkDown5', u'CPI', u'Unemployment']
	@return DataFrame with column ['Store','Dept','Date','IsHoliday',Feature1,Feature2...]
	'''
	return feature[['Store','Dept','Date','IsHoliday','Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment']]

def encoding(xs,_type='1-vs-all'):
	'''
	@param[in] xs iterable of value
	@return  DataFrame
	'''
	def _1_vs_all(x):
		v = DictVectorizer(sparse=False)
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
	months = map(lambda x:x.tm_mon,dates)
	names = map(lambda x:time.strftime('%Y-%m-%d',x),dates)
	return pd.DataFrame.from_dict({'Date':names,'WeekMonth':week_of_month,'WeekYear':week_of_year,'Month':months});

@decorators.disk_cached(utils.CACHE_DIR+'/date_mapping_002')
def date_mapping_002(f,t):
	mapping = date_mapping_001(f,t)
	df = mapping[['Date']]
	for key in filter(lambda f:f!='Date',mapping):
		encoded = encoding(mapping[key])
		buff = pd.DataFrame(encoded)
		buff.columns = map(lambda i:'%s_%d'%(key,i),range(len(buff.columns)))
		df = pd.merge(df,buff,left_index=True,right_index=True)
	return df

def feature_002(feature):
	'''
	@return id => (month,week)
	'''
	IDS = feature[['Store','Dept','Date','IsHoliday']]
	mapping = date_mapping_002('2010-01-01','2014-12-31')
	return pd.merge(IDS,mapping)

@decorators.disk_cached(utils.CACHE_DIR+'/id_mapping_001')
def id_mapping_001(name,max_id):
	encoded = encoding(range(max_id))
	stores = pd.DataFrame(encoded)
	stores.columns = map(lambda i:'%s_%d'%(name,i),range(len(stores.columns)))
	stores[name] = range(max_id)
	return stores

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

def feature_100(feature,keys,values,periods):
	'''
	@return { (keys+date) : [ sum(value1 in period1),sum(value1 in period2),sum(value1 in period3)...] }
	'''
	pass

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
			s    = Y[i]
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



if __name__ == '__main__':
	pass

