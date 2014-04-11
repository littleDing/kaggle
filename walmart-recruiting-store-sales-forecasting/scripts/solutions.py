from optparse import OptionParser
from multiprocessing import Pool
import random,logging,traceback,datetime,os,collections
import sklearn
import sklearn.linear_model
import sklearn.svm
import sklearn.ensemble
import pandas as pd

import utils,instances

class Predictor():
	'''
	build a predictor for every single id in ids
	'''
	def __init__(self,modelFactory,supportW=True,supportSparse=True,ids=None,negetiveY=None,tagging=None,num_core=1):
		self.ids = ids
		self.models = {}
		self.modelFactory = modelFactory
		self.supportW = supportW
		self.supportSparse = supportSparse
		self.negetiveY = negetiveY
		self.tagging = tagging
		self.num_core = num_core
	def split(self,ID):
		index = collections.defaultdict(list)
		for i in range(len(ID)):
			key = '-'.join(map(str,ID.loc[i][self.ids]))
			index[key].append(i)
		return index
	def fit(self,X,Y,W,index):
		def fit_instances(idx,tag):	
			model = self.modelFactory(tag=tag)
			if not self.supportW:
				idx = [ i for i in idx for j in range(int(W[i])) ]
			x,y,w = X[idx],Y[idx],W[idx]
			if not self.supportSparse:
				x = x.toarray()
			if self.supportW :
				model.fit(x,y,w)
			else :
				model.fit(x,y)
			return model
		self.models = {}
		log_step = len(index)/100+1
		def handle(arg):
			kid,key = arg
			idx = index[key]
			if self.negetiveY == 'ignore' :
				idx = filter(lambda i:Y[i]>0,idx)
			model = None
			if len(idx) >= 5 :
				model = fit_instances(idx,key)
			if kid % log_step==0:
				logging.info('model %s fit with #%d ins'%(key,len(idx)))
			return key,model
		if self.num_core >1:
			pool = Pool(self.num_core)
			models = pool.map(handle,enumerate(sorted(index.keys())))
			self.models = { key:model for key,model in models if model }
		else :
			for kid,key in enumerate(sorted(index.keys())):
				key,model = handle((kid,key))
				if model :
					self.models[key] = model
		return self
	def predict(self,X,index):
		Y = [0]*X.shape[0]
		log_step = len(index)/100+1
		for kid,key in enumerate(sorted(index.keys())):
			idx = index[key]
			if key not in self.models : 
				continue
			model = self.models[key]
			x = X[idx]
			if not self.supportSparse:
				x = x.toarray()
			y = model.predict(x)
			for i in xrange(len(idx)):
				Y[idx[i]] = y[i]
			if kid % log_step ==0 :
				logging.info('#%d of id %s predited'%(len(idx),key))
		return Y

class RGF():
	def __init__(self,reg_L2=1,tag='default',params='',lazy=True):
		self.tag = tag
		self.prefix = os.path.join(utils.TEMP_DIR,'rgf/',self.tag)
		self.params = 'reg_L2=%s'%(reg_L2)+params
		self.lazy = lazy
	def prepare_env(self,prefix,X,Y=None,W=None):
		'''
		write x,y,w to disk with prefix 
		'''
		def write(data,path):
			if data != None :
				df = pd.DataFrame(data)
				df.to_csv(path,sep=' ',header=False,index=False)
		write(X,prefix+'.x')
		write(Y,prefix+'.y')
		write(W,prefix+'.w')
	def find_model(self):
		path = None
		for i in range(100):
			fack = self.prefix+'.model-%02d'%i
			if os.path.exists(fack):
				path = fack
		return path
	def fit(self,X,Y,W):
		if self.lazy and self.find_model() :
			return self
		prefix = self.prefix + '.train'
		self.prepare_env(prefix,X,Y,W)
		params = '%s,train_x_fn=%s,test_x_fn=%s,train_y_fn=%s,test_y_fn=%s,train_w_fn=%s'%(self.params,prefix+'.x',prefix+'.x',prefix+'.y',prefix+'.y',prefix+'.w')
		params = params + ',model_fn_prefix=%s.model'%(self.prefix)
		cmd = '%s train_test %s'%(utils.CONFIGS['rgf'],params)
		ret = os.popen(cmd).read()
		#logging.info(ret)
		return self
	def predict(self,X):
		prefix = self.prefix + '.test'
		self.prepare_env(prefix,X)
		model  = self.find_model()
		params = 'model_fn=%s,test_x_fn=%s,prediction_fn=%s'%(model,prefix+'.x',prefix+'.pred')
		cmd = '%s predict %s'%(utils.CONFIGS['rgf'],params)
		try :
			ret = os.popen(cmd).read()
			yy = pd.read_csv(prefix + '.pred',header=None)[0]
		except Exception,e:
			logging.info(ret)
		return yy

def solution(train_path,test_path,
		modelFactory=sklearn.linear_model.LinearRegression,
		modelNeedID=None,
		featureFactory=instances.make_instance,
		feature=[],
		baseModelFactory=None,baseFeatureFactory=None,baseFeature=None,baseModelNeedID=None,
		version='current',**karg):
	
	def try_get_predict(model,modelNeedID,featureFactory,feature):
		if modelNeedID :
			test_x,test_y,test_w,test_ID,test_IDString,test_index = featureFactory(test_path,feature,modelNeedID)
		else :
			test_x,test_y,test_w,test_ID,test_IDString = featureFactory(test_path,feature)[:5]
		logging.info('testing data loaded!')
		if modelNeedID :
			yy = model.predict(test_x,test_index)
		else :
			yy = model.predict(test_x)
		return yy,test_IDString

	def try_get_model(modelFactory,modelNeedID,featureFactory,feature):
		if modelNeedID :
			train_x,train_y,train_w,train_ID,train_IDString,train_index = featureFactory(train_path,feature,modelNeedID)
		else :
			train_x,train_y,train_w,train_ID,train_IDString = featureFactory(train_path,feature)[:5]
		logging.info('training data loaded!')
		model = modelFactory()
		if modelNeedID :
			model = model.fit(train_x,train_y,train_w,train_index)
			logging.info('model fit!')
			yy = model.predict(train_x,train_index)
		else :
			model = model.fit(train_x,train_y,train_w)
			logging.info('model fit!')
			yy = model.predict(train_x)
		return model,train_y,train_w,yy

	model,train_y,train_w,yy = try_get_model(modelFactory,modelNeedID,featureFactory,feature)
	if baseModelFactory:
		baseModel,base_y,base_w,baseYY = try_get_model(baseModelFactory,baseModelNeedID,baseFeatureFactory,baseFeature)
		yy = [ y if abs(y)>0.1 else baseYY[i] for i,y in enumerate(yy) ]

	logging.info('wmae on training set : %s'%(utils.wmae(train_y,yy,train_w) ))
	
	yy,test_IDString = try_get_predict(model,modelNeedID,featureFactory,feature)
	if baseModelFactory:
		baseYY = try_get_predict(baseModel,baseModelNeedID,baseFeatureFactory,baseFeature)[0]
		yy = [ y if abs(y)>0.1 else baseYY[i] for i,y in enumerate(yy) ]
	
	ans = pd.DataFrame.from_dict({'Id':test_IDString,'Weekly_Sales':yy})
	ans.to_csv(os.path.join(utils.ANS_DIR,version+'.txt'),index=False)

solutions = {
	'20140314' : {
		'modelFactory' : ('sklearn.linear_model.LinearRegression',{}),
		'modelNeedID' : False,
		'train_path' : 'train.csv',
		'test_path' : 'test.csv',
		'feature' : [('001',),('002s',),('003s',),('006s',)],
		'featureFactory' : 'make_sparse_instance'
	},
	'LR' : {
		'modelFactory' : ('sklearn.linear_model.LinearRegression',{}),
		'modelNeedID' : False,
		'train_path' : 'train.csv',
		'test_path' : 'test.csv',
		'feature' : [('001f',),('002s',),('003s',),('004s',),
				('005s',),('006s',),
				('007s',),('008s',),
				('009s',),
		],
		'featureFactory' : 'make_sparse_instance'
	},
	'current' : {
		'train_path' : 'train.csv',
		'test_path' : 'test.csv',

		'modelFactory' : (
				'Predictor',
				{
					'ids' : ['Dept','Store'],
					#'modelFactory' : (lambda tag: sklearn.ensemble.GradientBoostingRegressor(loss='lad',n_estimators=50,max_depth=5)),
					'modelFactory' : (lambda tag:RGF(tag=tag,lazy=False)),
					'supportW' : True,
					'supportSparse' : False,
					'negetiveY' : 'ignore'
				}
		),
		'modelNeedID' : ['Dept','Store'],
		'feature' : [
				#('001_fna','002',{'n_estimators':800}),
				('001f',),
				('012',),
				('009',),
		],
		'featureFactory' : 'make_sparse_instance',

		'baseModelFactory' : None
		,'123':(
				'Predictor',
				{
					'ids' : ['Dept','Store'],
					'modelFactory' : (lambda tag: sklearn.ensemble.GradientBoostingRegressor(loss='lad',n_estimators=100,max_depth=5)),
					'supportW' : False,
					'supportSparse' : False,
					'negetiveY' : 'ignore'
				}
		),
		'baseModelNeedID' : ['Dept','Store'],
		'baseFeatureFactory' : 'make_sparse_instance',
		'baseFeature' : [
			('001_fna','002',{'n_estimators':800}),
			('012',),
			('009',),
		],
	},
}

def get_object(base,names):
	if len(names) == 0 :
		return base
	else :
		return get_object(base.__dict__[names[0]],names[1:])

def make_model_factory(name,karg):
	name = name.split('.')
	factory = get_object(globals()[name[0]],name[1:])
	return lambda : factory(**karg)

def run_solution(version='current'):
	karg = solutions[version]
	karg['version'] = version
	
	karg['modelFactory'] = make_model_factory(*karg['modelFactory'])
	karg['featureFactory'] = instances.__dict__[ karg['featureFactory'] ]
	
	if karg['baseModelFactory'] :
		karg['baseModelFactory'] = make_model_factory(*karg['baseModelFactory'])
		karg['baseFeatureFactory'] = instances.__dict__[ karg['baseFeatureFactory'] ]
	
	solution(**karg)

def main():
	parser = OptionParser()
	parser.add_option('', '--job', dest='job',help='jobname',default='run_solution');
	parser.add_option('', '--seed', dest='seed',type=int,default=123,help='random seed');
	parser.add_option('', '--options', dest='options',type=int,default=0
		,help='if set to 1, function will call with optparser\'s option as first argument');
	(options, args) = parser.parse_args()
	logging.warn((options,args))
	random.seed(options.seed)
	try:
		job = globals()[options.job]
		if options.options :
			job(options,*args)
		else :
			job(*args)
	except Exception,e:
		traceback.print_exc()
		logging.error(e)
		raise e

if __name__ == '__main__':
	main()
