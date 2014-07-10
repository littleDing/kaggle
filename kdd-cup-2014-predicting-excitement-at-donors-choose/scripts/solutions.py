from optparse import OptionParser
from multiprocessing import Pool
import random,logging,traceback,datetime,os,collections
import sklearn
import sklearn.linear_model
import sklearn.svm
import sklearn.ensemble
import sklearn.decomposition
from sklearn import svm,linear_model,ensemble,naive_bayes
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score
import pandas as pd,numpy as np

import utils,instances,models

def cross_validations(seed,fold,train_ID):
	'''
	@return [ (train_index,test_index) ]
	'''
	kfold = []
	for train_index,test_index in KFold(len(train_ID),fold,shuffle=False,random_state=seed):
		kfold.append( (train_index,test_index) )
	return kfold

def solution(cross_validation=None,
		modelFactory=sklearn.linear_model.LogisticRegression,
		featureFactory=instances.make_sparse_instance,
		feature=[],combination=0,transformer=None,
		train_dates=None,test_dates=None,
		version='current',**karg):
	def _train_test(train_x,train_y,test_x):
		logging.info('shapes : train_x=%s train_y=%s(+%s%%) test_x=%s'%(train_x.shape,train_y.shape,train_y.mean(),test_x.shape))
		model = modelFactory()							
		model.fit(train_x,train_y) 						
		train_yy = model.predict_proba(train_x)[:,1] 	
		train_auc = roc_auc_score(list(train_y),train_yy)		

		test_yy = model.predict_proba(test_x)[:,1]
		return test_yy,train_auc
	
	if combination==0:
		if train_dates == None and test_dates == None:
			train_X,train_Y,train_ID,test_X,test_ID = featureFactory(feature)
		else :
			train_X,train_Y,train_ID,test_X,test_ID = featureFactory(feature,train_dates,test_dates)
	else :
		train_X,train_Y,train_ID,test_X,test_ID = featureFactory(feature,combination)
	if transformer !=None:
		transformer.fit(train_X)
		train_X = transformer.transform(train_X)
		test_X = transformer.transform(test_X)

	logging.info('feature data loaded')	
	if cross_validation != None :
		seed,fold = cross_validation[:2]
		atmost = cross_validation[2] if len(cross_validation)>=3 else 9999
		logging.info('%s-fold cross_validating with seed %s'%(fold,seed))
		aucs = []
		for i,indexes in enumerate(cross_validations(seed,fold,train_ID)):
			logging.info('%d fold begins'%(i))
			train_index,test_index = indexes
			train_x,train_y = train_X[train_index],train_Y[train_index]
			test_x,test_y = train_X[test_index],train_Y[test_index]
			if test_y.sum()==0 or train_y.sum()==0 :
				logging.info('label not well seperated, give up this fold')
				continue

			test_yy,train_auc = _train_test(train_x,train_y,test_x)
			test_auc = roc_auc_score(list(test_y),test_yy)
			aucs.append( (train_auc,test_auc) )
			logging.info('%d fold finished #ins=%s,%s auc=%s,%s'%(i,len(train_y),len(test_y),train_auc,test_auc))
			
			atmost = atmost-1
			if atmost<=0 :
				break

		logging.warn('kf=%s\taucs=%s'%((seed,fold),aucs))
		acs = [ t0 for t0,t1 in aucs ]
		logging.warn('kf=%s\ttraining\tauc_min=%s\tauc_mean=%s\tauc_max=%s\tauc_std=%s'%(
					(seed,fold),	min(acs), 		np.mean(acs),max(acs),	np.std(acs) ) )
		acs = [ t1 for t0,t1 in aucs ]
		logging.warn('kf=%s\ttesting \tauc_min=%s\tauc_mean=%s\tauc_max=%s\tauc_std=%s'%(
					(seed,fold),	min(acs), 		np.mean(acs),max(acs),	np.std(acs) ) )

	test_yy,train_auc = _train_test(train_X,train_Y,test_X)
	logging.warn('train_auc=%s'%(train_auc))
	
	ans = pd.DataFrame.from_dict({'projectid':test_ID,'is_exciting':test_yy})
	mi,ma = ans.is_exciting.min(),ans.is_exciting.max()
	if ma>1 or mi<0:
		ans.is_exciting = (ans.is_exciting - mi)/(ma-mi)
	ans.to_csv(os.path.join(utils.ANS_DIR,version+'.txt'),index=False,cols=['projectid','is_exciting'])	

descrete_features = [('002',), ('003',), ('004a',), ('005', 10), ('006',),('008',),('007',) ]
combination_features  = [ descrete_features[:i+1] for i,f in enumerate(descrete_features) ]
combination_features += [ [f] for f in descrete_features ]
combination_features += [ [f1,f2] for i1,f1 in enumerate(descrete_features) for f2 in descrete_features[i1+1:] ]

solutions = {
	'current' : {
		'cross_validation' : (11717,14,3),
		'modelFactory' : ('naive_bayes.MultinomialNB',{}),
		#'modelFactory' : ('linear_model.LogisticRegression',{'penalty':'l1','C'11}),
		'feature' : [ ('001',),('002',),('003',),('004a',),('005',10),('006',),('008',),('009',)],
		#'feature' : [ ('001',),('002',),('003',),('004a',),('005',10),('006',)],
		'combination' : 0,
		'featureFactory' : 'make_sparse_instance'
	},
	'dense' : {
		'cross_validation' : (11717,14,3),
		'modelFactory' : ('ensemble.GradientBoostingClassifier',{'verbose':2,'max_features':'log2','n_estimators':400,'max_depth':5,'min_samples_leaf':2000}),
		'feature':[('001',),('002',),('004d',),('003dt',[u'is_exciting'],30),
			('003dtw',[u'is_exciting'],50,10),
			('003dtw',[u'is_exciting'],50,30),
			('003dtw',[u'is_exciting'],50,50),
		],
		'featureFactory' : 'make_dense_instance'
	},
	'dense2' : {
		'cross_validation' : (11717,14,3),
		'modelFactory' : ('ensemble.GradientBoostingClassifier',{'verbose':2,'n_estimators':100,'max_depth':5,'min_samples_leaf':2000}),
		'feature' : [ ('001',),('002',),('004d',),
			('007dt',[u'is_exciting', u'at_least_1_teacher_referred_donor', u'fully_funded', u'at_least_1_green_donation', u'great_chat', u'three_or_more_non_teacher_referred_donors', u'one_non_teacher_referred_donor_giving_100_plus', u'donation_from_thoughtful_donor', u'great_messages_proportion', u'teacher_referred_count', u'non_teacher_referred_count'],30),
			('007dtw',[u'is_exciting', u'at_least_1_teacher_referred_donor', u'fully_funded', u'at_least_1_green_donation', u'great_chat', u'three_or_more_non_teacher_referred_donors', u'one_non_teacher_referred_donor_giving_100_plus', u'donation_from_thoughtful_donor', u'great_messages_proportion', u'teacher_referred_count', u'non_teacher_referred_count'],30,10),
			],
		'featureFactory' : 'make_dense_instance'
	},
	'dense3' : {
		'cross_validation' : (11717,14,3),
		'modelFactory' : ('models.XGB',{'eval_metric':'auc','num_round':10,'nthread':1,'objective':'rank:pairwise',
			'bst:max_depth':5,'bst:min_child_weight':2000,'bst:subsample':1,'bst:eta':0.1}), 
		'feature' : [ #('002',),('004d',),('004d_1',),('006d',),('008d',),('020',),('021',),
			('050p',[ ('001',),('002',),('003',),('004a',)],),
			('050p',[ ('001',),('002',),('003',),('004a',),('005',10)],),
			('050p',[ ('001',),('002',),('003',),('004a',),('005',10),('006',)],),
			('050p',[ ('001',),('002',),('003',),('004a',),('005',10),('006',),('008',)],),
			('050p',[ ('001',),('002',),('003',),('004a',),('005',10),('006',),('008',),('009',)],),
		],
		'featureFactory' : 'make_dense_instance',
		'train_dates' : ('2010-01-01','2014-01-01'),
	},
	'dense7' : {
		'cross_validation' : (11717,14,14),
		'modelFactory' : ('models.XGB',{'eval_metric':'auc','num_round':10,'nthread':1,'objective':'rank:pairwise',
			'bst:max_depth':5,'bst:min_child_weight':2000,'bst:subsample':1,'bst:eta':0.1}), 
		'feature' : [ 
			('050pm',fs,'naive_bayes.BernoulliNB') for fs in combination_features
		],
		'featureFactory' : 'make_dense_instance',
		'train_dates' : ('2010-01-01','2014-01-01'),
		'test_dates'  : None,
	},
	'dense8' : {
		'cross_validation' : (11717,14,14),
		'modelFactory' : ('models.XGB',{'eval_metric':'auc','num_round':10,'nthread':1,'objective':'rank:pairwise',
			'bst:max_depth':5,'bst:min_child_weight':2000,'bst:subsample':1,'bst:eta':0.1}), 
		'feature' : [ 
			('050pm',[('007',)],'naive_bayes.BernoulliNB')   
		], 
		'featureFactory' : 'make_dense_instance',
		'train_dates' : ('2010-01-01','2014-01-01'),
		'test_dates'  : None,
	},
	'dense105' : {
		'cross_validation' : (11717,14,14),
		'modelFactory' : ('models.XGB',{'eval_metric':'auc','num_round':400,'nthread':6,'objective':'rank:pairwise',
			'bst:max_depth':5,'bst:min_child_weight':2000,'bst:subsample':1,'bst:eta':0.1}), 
		'feature' : [ ('002',),('004d',),('004d_1',),('006d',),('008d',),('020',),('021',),
			#('040h','essay'),('040h','need_statement'),('040h','short_description'),('040h','title'),
			('017',[10],[90,180,360]),
			#('030_1',10),('030_2',10,[10],[90,180,360]),
			('007dtna_1',[u'is_exciting', u'at_least_1_teacher_referred_donor', u'fully_funded', u'at_least_1_green_donation', u'great_chat', u'three_or_more_non_teacher_referred_donors', u'one_non_teacher_referred_donor_giving_100_plus', u'donation_from_thoughtful_donor', u'great_messages_proportion', u'teacher_referred_count', u'non_teacher_referred_count']),],
		'train_dates' : ('2010-01-01','2014-01-01'),
		'featureFactory' : 'make_dense_instance'
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

	logging.info(karg)
	
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
