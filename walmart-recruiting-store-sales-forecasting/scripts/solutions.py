from optparse import OptionParser
import random,logging,traceback,datetime,os
import sklearn
import sklearn.linear_model
import sklearn.svm
import pandas as pd

import utils,instances

class Predictor():
	def __init__(self,ids):
		pass
	def split(self,ID):
		pass
	def fit(self,X,Y,W,ID):
		pass
	def predict(self,X,ID):
		pass


def solution(train_path,test_path,modelFactory=sklearn.linear_model.LinearRegression,feature=[],version='current'):
	train_x,train_y,train_w,train_ID,train_IDString = instances.make_instance(train_path,feature)
	
	model = modelFactory()
	model = model.fit(train_x,train_y,train_w)
	yy = model.predict(train_x)
	logging.info('wmae on training set : %s'%(utils.wmae(train_y,yy,train_w) ))
	
	test_x,test_y,test_w,test_ID,test_IDString = instances.make_instance(test_path,feature)
	yy = model.predict(test_x)
	ans = pd.DataFrame.from_dict({'Id':test_IDString,'Weekly_Sales':yy})
	ans.to_csv(os.path.join(utils.ANS_DIR,version+'.txt'),index=False)

solutions = {
	'current' : {
		'modelFactory' : ('sklearn.svm.SVR',{'kernel':'poly','C':100}), 
		'train_path' : 'train.csv',
		'test_path' : 'test.csv',
		'feature' : [('001',),('002',),('003',)],
	}
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
	solution(**karg)

def main():
	parser = OptionParser()
	parser.add_option('', '--job', dest='job',help='jobname');
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