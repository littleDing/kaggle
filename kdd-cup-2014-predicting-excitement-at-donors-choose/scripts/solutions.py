from optparse import OptionParser
from multiprocessing import Pool
import random,logging,traceback,datetime,os,collections
import sklearn
import sklearn.linear_model
import sklearn.svm
import sklearn.ensemble
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score
import pandas as pd,numpy as np

import utils,instances

class RGF():
	def __init__(self,reg_L2=1,tag='default',params='',lazy=True,sparse=False):
		self.tag = tag
		self.prefix = os.path.join(utils.TEMP_DIR,'rgf/',self.tag)
		self.params = 'reg_L2=%s'%(reg_L2)+params
		self.lazy = lazy
		self.sparse = sparse
	def prepare_env(self,prefix,X,Y=None,W=None):
		'''
		write x,y,w to disk with prefix 
		'''
		def write(data,path):
			if data != None :
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
	def fit(self,X,Y,W):
		if self.lazy and self.find_model() :
			return self
		prefix = self.prefix + '.train'
		self.prepare_env(prefix,X,Y,W)
		params = '%s,train_x_fn=%s,test_x_fn=%s,train_y_fn=%s,test_y_fn=%s,train_w_fn=%s'%(self.params,prefix+'.x',prefix+'.x',prefix+'.y',prefix+'.y',prefix+'.w')
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

def cross_validations(seed,fold,train_ID):
	'''
	@return [ (train_index,test_index) ]
	'''
	kfold = []
	for train_index,test_index in KFold(len(train_ID),fold,shuffle=True,random_state=seed):
		kfold.append( (train_index,test_index) )
	return kfold

def solution(cross_validation=None,
		modelFactory=sklearn.linear_model.LogisticRegression,
		featureFactory=instances.make_sparse_instance,
		feature=[],
		version='current',**karg):
	def _train_test(train_x,train_y,test_x):
		logging.info('shapes : train_x=%s train_y=%s test_x=%s'%(train_x.shape,train_y.shape,test_x.shape))
		model = modelFactory()							
		model.fit(train_x,train_y) 						
		train_yy = model.predict_proba(train_x)[:,1] 	
		train_auc = roc_auc_score(train_y,train_yy)		

		test_yy = model.predict_proba(test_x)[:,1]
		return test_yy,train_auc

	train_X,train_Y,train_ID,test_X,test_ID = featureFactory(feature)
	logging.info('feature data loaded')	
	if cross_validation != None :
		seed,fold = cross_validation
		logging.info('%s-fold cross_validating with seed %s'%(fold,seed))
		aucs = []
		for i,indexes in enumerate(cross_validations(seed,fold,train_ID)):
			logging.info('%d fold begins'%(i))
			train_index,test_index = indexes
			train_x,train_y = train_X[train_index],train_Y[train_index]
			test_x,test_y = train_X[test_index],train_Y[test_index]

			test_yy,train_auc = _train_test(train_x,train_y,test_x)
			test_auc = roc_auc_score(test_y,test_yy)
			aucs.append( (train_auc,test_auc) )
			logging.info('%d fold finished #ins=%s,%s auc=%s,%s'%(i,len(train_y),len(test_y),train_auc,test_auc))
		logging.warn('aucs=%s')
		acs = [ t0 for t0,t1 in aucs ]
		logging.warn('kf=%s\ttraining\tauc_min=%s\tauc_mean=%s\tauc_max=%s\tauc_std=%s'%(
					(seed,fold),	min(acs), 		np.mean(acs),max(acs),	np.std(acs) ) )
		acs = [ t1 for t0,t1 in aucs ]
		logging.warn('kf=%s\ttesting \tauc_min=%s\tauc_mean=%s\tauc_max=%s\tauc_std=%s'%(
					(seed,fold),	min(acs), 		np.mean(acs),max(acs),	np.std(acs) ) )

	test_yy,train_auc = _train_test(train_X,train_Y,test_X)
	logging.warn('train_auc=%s'%(train_auc))
	
	ans = pd.DataFrame.from_dict({'projectid':test_ID,'is_exciting':test_yy})
	ans.to_csv(os.path.join(utils.ANS_DIR,version+'.txt'),index=False,cols=['projectid','is_exciting'])	
	
solutions = {
	'current' : {
		'cross_validation' : (123,5),
		'modelFactory' : ('sklearn.linear_model.LogisticRegression',{'penalty':'l1'}),
		'feature' : [ ('001',),('002',),('003',),('004',)],
		'featureFactory' : 'make_sparse_instance'
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
