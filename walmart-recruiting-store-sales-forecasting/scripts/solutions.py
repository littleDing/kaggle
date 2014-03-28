from optparse import OptionParser
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
	def __init__(self,ids,modelFactory,supportW=True,supportSparse=True):
		self.ids = ids
		self.models = {}
		self.modelFactory = modelFactory
		self.supportW = supportW
		self.supportSparse = supportSparse
	def split(self,ID):
		index = collections.defaultdict(list)
		for i in range(len(ID)):
			key = '-'.join(map(str,ID.loc[i][self.ids]))
			index[key].append(i)
		return index
	def fit(self,X,Y,W,ID):
		index = self.split(ID)
		for key in sorted(index.keys()):
			idx = index[key]
			model = self.modelFactory()    #sklearn.linear_model.LinearRegression()
			if not self.supportW:
				idx = [ i for i in idx for j in range(int(W[i])) ]
			x,y,w = X[idx],Y[idx],W[idx]
			if not self.supportSparse:
				x = x.toarray()
			if self.supportW :
				model.fit(x,y,w)
			else :
				model.fit(x,y)
			self.models[key] = model
			logging.info('model %s fit with #%d ins'%(key,len(idx)))
		return self
	def predict(self,X,ID):
		index = self.split(ID)
		Y = [0]*len(ID)
		for key in sorted(index.keys()):
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
			logging.info('#%d of id %s predited'%(len(idx),key))
		return Y

def solution(train_path,test_path,
		modelFactory=sklearn.linear_model.LinearRegression,
		modelNeedID=False,
		featureFactory=instances.make_instance,
		feature=[],version='current'):
	train_x,train_y,train_w,train_ID,train_IDString = featureFactory(train_path,feature)
	logging.info('training data loaded!')
	model = modelFactory()
	if modelNeedID :
		model = model.fit(train_x,train_y,train_w,train_ID)
		logging.info('model fit!')
		yy = model.predict(train_x,train_ID)
	else :
		model = model.fit(train_x,train_y,train_w)
		logging.info('model fit!')
		yy = model.predict(train_x)
	logging.info('wmae on training set : %s'%(utils.wmae(train_y,yy,train_w) ))
	
	test_x,test_y,test_w,test_ID,test_IDString = featureFactory(test_path,feature)
	logging.info('testing data loaded!')
	if modelNeedID :
		yy = model.predict(test_x,test_ID)
	else :
		yy = model.predict(test_x)
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
		'modelFactory' : (
				'Predictor',
				{
					'ids' : ['Dept','Store'],
					'modelFactory' : sklearn.linear_model.LinearRegression,
					#'modelFactory' :(lambda :sklearn.ensemble.GradientBoostingRegressor(n_estimators=100,max_depth=5)),
					'supportW' : True,
					'supportSparse' : True,
				}
		),
		'modelNeedID' : True,
		'train_path' : 'train.csv',
		'test_path' : 'test.csv',
		'feature' : [('001f',),('002s',),('003s',),
				#('005Ss',),('006Ss',),
				('008s',),
		],
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
