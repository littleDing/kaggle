import nltk,sys,logging,re,csv,multiprocessing,itertools,traceback,numpy as np
from collections import defaultdict,Counter
from tools import *
from conf import *
from DiskCache import *
logging.basicConfig(level=logging.WARN,format='%(asctime)s %(funcName)s@%(filename)s#%(lineno)d %(levelname)s %(message)s')
def helloworld(options,*args):
	@diskcached('just_test')
	def hehe():
		return args
	kaka = hehe()
	logging.warn([kaka,options,args])

def state_file_20131026(options,path,*args):
	filename=getFilename(path)
	@diskcached(TMP_DIR+filename)
	def do_stats():
		reader=wrap_line_logger(csv.reader(open(path)),interval=10000)
		reader.next()
		lines=0
		words_title={}
		words_body={}
		tags=defaultdict(lambda :[0,0,0,0])
		tags_in_title=0
		tags_in_body=0
		tags_in_both=0
		tags_cnt = [0] * 10
		for tid,title,body,tag in reader:
			lines+=1
			tag=tag.split()
			for t in tag :
				tags[t][0]+=1
				both = 0
				if title.find(t) != -1:
					tags_in_title +=1
					tags[t][1] +=1
					both +=1
				if body.find(t) != -1:
					tags_in_body +=1
					tags[t][2] +=1
					both +=1
				if both :
					tags[t][3] +=1
					tags_in_both  +=1
			tags_cnt[len(tag)] +=1
		return lines,words_title,words_body,dict(tags),tags_in_title,tags_in_body,tags_in_both,tags_cnt
	lines,words_title,words_body,tags,tags_in_title,tags_in_body,tags_in_both,tags_cnt = do_stats()
	print lines,len(tags),sum([c*t for c,t in enumerate(tags_cnt)]),tags_in_title,tags_in_body,tags_in_both,map(lambda x:x*1.0/lines,tags_cnt),tags_cnt
	with open(TMP_DIR+filename+'.tagcounts','w') as fout:
		writer = csv.writer(fout)
		writer.writerows([ [tag]+cnt for tag,cnt in tags.iteritems()])

db_20131027=Database(conf.PICKLE_DIR+'db_20131027/',False)
#@dbcached(db_20131027)
def csv_data(path):
	reader=wrap_line_logger(csv.reader(open(path)),name='csv_data:'+path,interval=10000)
	reader.next()
	for line in reader:
		yield line

stopwords=set(nltk.corpus.stopwords.words('english'))
def map_raw_to_obj_20131027(row):
	row[1] = [w for w in nltk.word_tokenize(row[1]) if not w in stopwords ] 
	row[2] = [w for w in nltk.word_tokenize(row[2]) if not w in stopwords ]
	if len(row)>3:
		row[3]=row[3].split()
	return row
@dbcached(db_20131027)
def towords_20131027(path):
	pool=multiprocessing.Pool(4)
	for row in wrap_line_logger(pool.imap(map_raw_to_obj_20131027,csv_data(path),32),name='towords_20131027:'+path,interval=10000):
		yield row
#striper  = MLStripper(['code'])
def map_raw_to_obj_20131030(row):
	def do_map(idx):
		#striper.clear()
		#striper.feed(row[idx].decode('utf-8','ignore'))
		#row[idx] = striper.get_data().lower()
		row[idx] = strip_tags(row[idx].lower(),['code','blockquote'])
		row[idx] = [ w for w in nltk.word_tokenize(row[idx]) if not w in stopwords ]
		row[idx] = [ w for w in row[idx] if re.match('.*[a-z].*',w) ]
	do_map(1)
	do_map(2)
	if len(row)>3:
		row[3]=row[3].split()
	return row
@dbcached(db_20131027)
def towords_20131030(path):
	'''	output format [id,[title_words],[body_words],tags]'''
	pool=multiprocessing.Pool(4)
	for row in wrap_line_logger(pool.imap(map_raw_to_obj_20131030,csv_data(path),1024),name='towords_20131030:'+path,interval=10000):
		yield row

from gensim import corpora, models, similarities
@dbcached(db_20131027,'object')
def dictionary_20131029(paths):
	text_dict = corpora.Dictionary()
	tag_dict  = corpora.Dictionary()
	for path in paths:
		i=0
		for row in wrap_line_logger(towords_20131027(path),name='dictionary_20131029:'+path,interval=10000):
			text_dict.add_documents(row[1:3])
			tag_dict.add_documents(row[3:])
			i=i+1
			if i %10000 == 0 :
				logging.warn(text_dict)
				logging.warn(tag_dict)
	text_dict.filter_extremes(10,0.5,None)
	text_dict.merge_with(tag_dict)
	return text_dict
@dbcached(db_20131027,'object')
def dictionary_20131030(paths):
	text_dict = corpora.Dictionary()
	tag_dict  = corpora.Dictionary()
	for path in paths:
		i=0
		for row in wrap_line_logger(towords_20131030(path),name='dictionary_20131030:'+path,interval=10000):
			text_dict.add_documents(row[1:3])
			tag_dict.add_documents(row[3:])
			i=i+1
			if i %10000 == 0 :
				logging.warn(text_dict)
				logging.warn(tag_dict)
	text_dict.filter_extremes(10,1.0,None)
	text_dict.merge_with(tag_dict)
	return text_dict

towordid_20131029_map_dict=None
towordid_20131029_map_rows=None
def towordid_20131029_map(ins):
	return [ins[0],reduce((lambda x,y:x+y),[ towordid_20131029_map_dict.doc2bow(ins[r]) for r in towordid_20131029_map_rows ],[])]	
@dbcached(db_20131027)
def towordid_20131029(paths,path,rows):
	'''	output [id,[(wordid,cnt)]] '''
	global towordid_20131029_map_dict,towordid_20131029_map_rows
	towordid_20131029_map_dict = dictionary_20131030([options.test,options.train])
	towordid_20131029_map_rows = rows
	pool=multiprocessing.Pool(3)
	for ins in wrap_line_logger(pool.imap(towordid_20131029_map,towords_20131030(path),1024),name='towordid:'+path,interval=10000):
		yield ins

@dbcached(db_20131027,'object')
def tfidf_model_20131029(paths,rows):
	def load_corpus():
		for path in paths:
			for did,doc in towordid_20131029(paths,path,rows):
				yield doc
	corpus = wrap_line_logger(load_corpus(),name='tfidf_model:'+str(paths),interval=10000)
	return models.TfidfModel(corpus)

@dbcached(db_20131027)
def totfidf_20131029(model_paths,model_rows,path,rows):
	'''output format : [id,[(wordid,tfidf)]]'''
	model = tfidf_model_20131029(model_paths,model_rows)
	for did,doc in wrap_line_logger(towordid_20131029(model_paths,path,rows),name='tfidf:'+path+str(rows)):
		yield [did,model[doc]]
	return 

def load_tf_idf_20131026(options):
	rows = [1,2]
	paths = [ options.train,options.test ]
	dic = dictionary_20131030(paths)
	model = tfidf_model_20131029(paths,rows)
	for doc in totfidf_20131029(paths,rows,options.path,map(int,options.rows.split(","))):
		pass
	return 
	for rs in [ [2],[1,2] ]:
		for path in paths:
			for doc in totfidf_20131029(paths,rows,path,rs):
				pass
	return 

def solution_20131030(options):
	''' if any toptags appear, pick it, then fill the left space with top words in text '''
	topwords = {}
	with open(DATA_DIR + 'topwords.txt') as fin:
		for line in fin:
			sp = line.split(",")
			topwords[sp[0]]=int(sp[1])	
	with open(TMP_DIR+'solution_20131030.txt','w') as fout:
		fout.write('"Id","Tags"'+"\n")
		for ins in wrap_line_logger(towords_20131027(DATA_DIR+'Test.csv'),name='solution_20131030',interval=100000):
			l = len(ins[1]) + len(ins[2]) + 100
			words = Counter(ins[1]+ins[2])
			for w in words :
				if w in topwords:
					words[w] += l + topwords[w]
			fout.write(ins[0]+',"'+ ' '.join(map(lambda x:x[0],words.most_common(5)))  +'"\n')
def solution_20131031(options):
	''' from solution_20131030, remove the ones never be in Train.csv's tags '''
	topwords = {}
	with open(DATA_DIR + 'topwords.txt') as fin:
		for line in fin:
			sp = line.split(",")
			topwords[sp[0]]=int(sp[1])	
	with open(TMP_DIR+'solution_20131031.txt','w') as fout:
		fout.write('"Id","Tags"'+"\n")
		with open(TMP_DIR+'solution_20131030.txt') as fin:
			reader = csv.reader(fin)
			reader.next()
			for key,tags in reader:
				tags = [tag for tag in tags.split() if tag in topwords]
				fout.write(key+',"'+' '.join( tags  )+'"\n' )

def submition_to_path(path):
	def make_func(func):
		def _func(*args,**kargs):
			with open(path,'w') as fout:
				fout.write('"Id","Tags"'+"\n")
				for key,tags in func(*args,**kargs):
					fout.write(key+',"'+' '.join(tags)+'"\n' )
		return _func
	return make_func


## features should output [  [ id,[ wordid, features ] ] ]
def feature_20131102(path):
	''' 
		[[doc,[(wordid,tfidf)]]] => [docid,{ wordid : [tfidf_all,tfidf_title,tfidf_body] } ]
	'''
	paths = [DATA_DIR+'Train.csv',DATA_DIR+'Test.csv']
	tfidfs = [ totfidf_20131029(paths,[1,2],path,r) for r in [ [1,2],[1],[2] ] ]
	for docs in itertools.izip(*tfidfs):
		did = docs[0][0]
		words = defaultdict(lambda :[0,0,0])
		for i,doc in enumerate(docs):
			for wid,tfidf in doc[1]:
				words[wid][i] = tfidf
		yield did,words
import sklearn,scipy
from sklearn import *
from scipy import *
def solution_20131102(options,path):
	@dbcached(db_20131027)
	def solution_20131102_features(fpath,withTags):
		dictionary = dictionary_20131030([options.test,options.train])
		features = feature_20131102(fpath)
		for doc in wrap_line_logger(csv_data(fpath),name='solution_20131102_feature'+fpath,interval=100000):
			feas = features.next()
			tags = []
			did = doc[0]
			if feas[0] == did :
				if len(feas[1])<=0:
					logging.warn(feas)
					logging.warn(doc)
				if withTags:
					tags = [ dictionary.token2id.get(t,-1) for t in doc[3].split()]
				for wid,f in feas[1].iteritems():
					y = 1 if wid in tags else 0
					yield did,y,[ wid+1000 ] + [ 0,1,2 ],[1]+f
			else :
				print 'sth wrong!'
				print feas
				print doc
				raise Exception('doc id not match!')
#	withTags = True if path==options.train else False
#	for doc in solution_20131102_features(path,withTags):
#		pass
	
	@dbcached(db_20131027,'object')
	def solution_20131102_models_at_batch(batch,loss='log',penalty='l2'):
		raise 'load me first!'
	def load_model_batches(atmost=10000000,loss='log',penalty='l2'):
		model = linear_model.SGDClassifier(loss=loss,penalty=penalty)
		model.classes_ = np.array([0,1])
		x,ys = [],[]
		@dbcached(db_20131027,'object')
		def solution_20131102_models_at_batch(batch,loss='log',penalty='l2'):
			model.partial_fit(x,ys)
			return model
		batches = itertools.groupby(enumerate(solution_20131102_features(options.train,True)),lambda x:x[0]/1000000)
		for batch,values in wrap_line_logger(batches,name='solution_20131102_model:train_batch:',interval=1):
			ys,xidx,xvalues = [],[[],[]],[]
			for i,value in values:
				did,y,xks,xvs = value
				xidx[0] += [len(ys)] * len(xks)
				xidx[1] += xks
				xvalues += xvs
				ys.append(y)
			x = scipy.sparse.csr_matrix((xvalues,xidx))
			solution_20131102_models_at_batch(batch,loss,penalty)
			if batch >= atmost:
				break
#	load_model_batches(10000,options.loss,options.penalty)
	def make_train_preds(atmost=100000,batch=100,loss='log',penalty='l2'):
		def _make_train_preds():
			model = solution_20131102_models_at_batch(batch,loss,penalty)
			dic = dictionary_20131030([options.test,options.train]).token2id
			dic = { wid:w for w,wid in dic.iteritems() }
			cnt = 0
			dim = len(model.coef_[0])
			for did,values in wrap_line_logger(itertools.groupby( solution_20131102_features(path,True if path==options.train else False),lambda x:x[0]),interval=10000):
				words = []
				ys,xidx,xvalues = [],[[],[]],[]
				for did,y,xks,xvs in values:
					ever = False
					for i in xrange(len(xks)):
						if xks[i] < dim:
							xidx[0].append(len(ys))
							xidx[1].append(xks[i])
							xvalues.append(xvs[i])
							ever=True
					if ever: 
						ys.append(y)
						words.append(dic[xks[0]-1000])
				x = scipy.sparse.csr_matrix((xvalues,xidx),shape=[len(ys),len(model.coef_[0])])
				y = model.predict_proba(x)
				scores = sorted([ (y[i][1],words[i],ys[i]) for i in range(len(y)) ],reverse=True)
				yield did,scores #  [ '%s:%s:%s'%(w,y,yy) for y,w,yy in scores ]
				cnt +=1
				if cnt >= atmost:
					break
		@submition_to_path(TMP_DIR+'solution_20131102_%s_%s_%s'%(batch,loss,penalty))
		def write_preds():
			for did,scores in _make_train_preds():
				yield did,[ w for y,w,yy in scores[:5] ]
		write_preds()
	
	make_train_preds(10000*10000,options.batch,options.loss,options.penalty)


@submition_to_path(TMP_DIR+'std_solution')
def std_solution(options):
	for ins in wrap_line_logger(csv_data(DATA_DIR+'Train.csv'),name='std_solution'):
		yield ins[0],ins[3].split()

def normalize(options,path):
	std = csv.reader(open(TMP_DIR+'solution_20131030.txt'))
	source = csv.reader(open(path))
	std.next()
	source.next()
	@submition_to_path(path+'.normalized.txt')
	def _solve():
		for line in source:
			s = std.next()
			while s[0]!=line[0]:
				yield s[0],[s[1]]
				s = std.next()
			yield line[0],[line[1]]
	_solve()

def judge(options,path):
	cnt = 0
	score = 0
	std = csv.reader(open(options.std))
	std.next()
	with open(path) as fin:
		reader = csv.reader(fin)
		reader.next()
		for key,tags in wrap_line_logger(reader,name="judge_of_"+path):
			tags = tags.split()
			cnt +=1
			std_tags = []
			for k,tgs in std:
				if k == key :
					std_tags = tgs.split()
					break
			if len(std_tags) > 0:
				common = len([t for t in std_tags if t in tags])*1.0
				p = common / len(tags)
				r = common /len(std_tags)
				score += 2*p*r/(p+r)
	print score/cnt

def submit_at(options,path,k):
	k=int(k)
	@submition_to_path(path+'.at%d.txt'%(k))
	def run():
		reader=csv.reader(open(path))
		reader.next()
		for did,words in reader:
			words = words.split()
			yield did,words[:k]
	run()

from optparse import OptionParser
import random

def main():
	parser = OptionParser()
	parser.add_option('', '--job', dest='job',help='jobname');
	parser.add_option('', '--seed', dest='seed',type=int,default=123,help='random seed');
	parser.add_option('', '--train', dest='train',type=str,default=DATA_DIR+'Train.csv',help='Train data');
	parser.add_option('', '--test', dest='test',type=str,default=DATA_DIR+'Test.csv',help='Test data');
	parser.add_option('', '--std', dest='std',type=str,default=TMP_DIR+'std_solution',help='standard solution for judge');
	parser.add_option('', '--rows', dest='rows',type=str,help='rows to handle');
	parser.add_option('', '--path', dest='path',type=str,help='file path to handle');
	parser.add_option('', '--loss', dest='loss',type=str,help='loss function');
	parser.add_option('', '--penalty', dest='penalty',type=str,help='penalty type');
	parser.add_option('', '--batch', dest='batch',type=int,help='model batch');
	global options
	(options, args) = parser.parse_args()
	logging.warn((options,args))
	random.seed(options.seed)
	try:
		globals()[options.job](options,*args)
	except Exception,e:
		traceback.print_exc()
		print e

if __name__ == '__main__':
	main()
