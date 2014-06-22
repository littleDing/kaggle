import numpy as np
import utils

class RGF():
	def __init__(self,reg_L2=1,tag='default',params='',lazy=True,sparse=False):
		self.tag = tag
		self.prefix = os.path.join(utils.RGF_TEMP_DIR,self.tag)
		self.params = 'reg_L2=%s,'%(reg_L2)+params
		self.lazy = lazy
		self.sparse = sparse
	def prepare_env(self,prefix,X,Y=None,W=None):
		'''
		write x,y,w to disk with prefix 
		'''
		def write(data,path):
			if type(data) != type(None) :
				df = pd.DataFrame(data)
				df.to_csv(path,sep=' ',header=False,index=False)
		if self.sparse :
			X = X.toarray()
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
	def fit(self,X,Y,W=None):
		if self.lazy and self.find_model() :
			return self
		prefix = self.prefix + '.train'
		self.prepare_env(prefix,X,Y,W)
		params = '%s,train_x_fn=%s,test_x_fn=%s,train_y_fn=%s,test_y_fn=%s'%(self.params,prefix+'.x',prefix+'.x',prefix+'.y',prefix+'.y')
		if W != None:
			params = params + ',train_w_fn=%s'%(prefix+'.w')
		params = params + ',model_fn_prefix=%s.model'%(self.prefix)
		cmd = '%s train_test %s'%(utils.CONFIGS['rgf'],params)
		ret = os.popen(cmd).read()
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
	def predict_proba(self,X):
		return self.predict(X)

import xgboost as xgb

class XGB():
	def __init__(self,num_round=100,**karg):
		self.param = karg
		self.num_round = num_round
	def fit(self,X,Y):
		data = xgb.DMatrix(X)
		data.set_label(Y)
		data.set_group([len(Y)])
		evallist  = [(data,'train')]
		self.model = xgb.train(self.param,data,self.num_round,evallist)
	def predict(self,X):
		data = xgb.DMatrix(X)
		self.model.predict(data)
		return data.get_label()
	def predict_proba(self,X):
		data = xgb.DMatrix(X)
		yy = np.zeros((X.shape[0],2))
		yy[:,1] = self.model.predict(data)
		yy[:,0] = 1-yy[:,1]
		return yy
		

