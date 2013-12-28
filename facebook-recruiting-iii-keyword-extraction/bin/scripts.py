import gensim,nltk,sys,logging,re,csv,multiprocessing,itertools,traceback,numpy as np
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
	reader=wrap_line_logger(csv.reader(open(path)),name='csv_data:'+path,interval=100000)
	reader.next()
	for line in reader:
		yield line

def state_titles(options):
	for path in [options.train,options.test]:
		with open(path+'.title','w') as fout:
			for line in csv_data(path):
				fout.write(line[0]+'\t'+line[1]+'\n')
def state_titles_tag(options):
	for path in [options.train]:
		with open(path+'.title_tag','w') as fout:
			for line in csv_data(path):
				toprints = [line[0],line[1],line[3]]
				fout.write('\t'.join(toprints)+'\n')

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
	'''	output format [id,[title_words],[body_words],[tags]]'''
	pool=multiprocessing.Pool(4)
	for row in wrap_line_logger(pool.imap(map_raw_to_obj_20131030,csv_data(path),1024),name='towords_20131030:'+path,interval=10000):
		yield row

def map_raw_to_obj_20131126(row):
	def do_map(idx):
		row[idx] = strip_tags(row[idx].lower(),['code','blockquote'])
		row[idx] = [ w for w in nltk.word_tokenize(row[idx]) if not w in stopwords ]
		row[idx] = [ w for w in row[idx] if re.match('.*[a-z].*',w) ]
	do_map(1)
	do_map(2)
	if len(row)>3:
		row[3]=row[3].split()
	return row
@dbcached(db_20131027)
def towords_20131126(path):
	'''	output format [id,[title_words],[body_words],[tags]]'''
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

@dbcached(db_20131027,'object')
def word2vec_20131117(paths,rows,size):
	dic = dictionary_20131030(paths)
	words = dic.token2id
	def make_sentences():
		for path in paths:
			for doc in wrap_line_logger(towords_20131030(path),name='word2vec_20131117:%s:%s'%(path,rows)):
				yield reduce((lambda x,y:x+y),[ [ w for w in doc[r] if w in words ] for r in rows ],[])
	model=gensim.models.Word2Vec(make_sentences(),size=size)
	return model

def load_word2vec_20131117(options,paths,rows,size):
	paths= paths.split(',')
	rows = map(int,rows.split(','))
	size = int(size)
	model = word2vec_20131117(paths,rows,size)

@dbcached(db_20131027,'object')
def topwords_20131120(paths,rows,size,topn):
	model = word2vec_20131117(paths,rows,size)
	return { w:model.most_similar([w],topn=topn) for w in wrap_line_logger(model.vocab.iterkeys(),name='topwords_20131120:%s:%s:%s:%s'%(paths,rows,size,topn),interval=10000) }

def load_topwords(options,paths,rows,size,topn):
	paths= paths.split(',')
	rows = map(int,rows.split(','))
	size = int(size)
	topn = int(topn)
	model = topwords_20131120(paths,rows,size,topn)

towordid_20131029_map_dict=None
towordid_20131029_map_rows=None
def towordid_20131029_map(ins):
	return [ins[0],reduce((lambda x,y:x+y),[ towordid_20131029_map_dict.doc2bow(ins[r]) for r in towordid_20131029_map_rows ],[])]	
@dbcached(db_20131027)
def towordid_20131029(path,rows):
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
			for did,doc in towordid_20131029(path,rows):
				yield doc
	corpus = wrap_line_logger(load_corpus(),name='tfidf_model:'+str(paths),interval=10000)
	return models.TfidfModel(corpus)
@dbcached(db_20131027,'object',( (lambda s:models.LsiModel.load(s)) , (lambda x,s:x.save(s)) )  )
def lsi_model_20131127(paths,rows):
	def load_corpus():
		for path in paths:
			for did,doc in towordid_20131029(path,rows):
				yield int(did),doc
	id2word = { wid:word for word,wid in dictionary_20131030(paths).token2id.items() }
	corpus = wrap_line_logger(load_corpus(),name='lsi_model:'+str(paths),interval=10000)
	model = models.LsiModel(id2word=id2word) 
	for key,docs in itertools.groupby(corpus,lambda x:x[0]/1000):
		logging.warn('training on %s'%(docs))
		model.add_documents([ doc for did,doc in docs ])
	return model
def load_lsi_model(options,paths,rows):
	paths = paths.split(',')
	rows  = map(int,rows.split(','))
	model = lsi_model_20131127(paths,rows)

@dbcached(db_20131027)
def tolsi_20131127(model_paths,path,rows):
	'''output format : [id,[(topicid,w)]]'''
	model = lsi_model_20131127(model_paths,rows)
	for did,doc in wrap_line_logger(towordid_20131029(path,rows),name='tolsi:'+path+str(rows)):
		yield [did,model[doc]]
def load_tolsi(options,path,rows):
	paths = [options.test,options.train]           #paths.split(',')
	rows = map(int,rows.split(','))
	for did,doc in tolsi_20131127(paths,path,rows):
		pass

@dbcached(db_20131027)
def totfidf_20131029(model_paths,model_rows,path,rows):
	'''output format : [id,[(wordid,tfidf)]]'''
	model = tfidf_model_20131029(model_paths,model_rows)
	for did,doc in wrap_line_logger(towordid_20131029(model_paths,path,rows),name='tfidf:'+path+str(rows)):
		yield [did,model[doc]]

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
@dbcached(db_20131027,'object')
def wid_counts_20131115(paths,path):
	''' { wordid : [#being_tag,#in_title,#in_body,sum(#tags/#words_title),sum(#tags/#words_body) ] }'''
	words = defaultdict(lambda :[0,0,0,0,0])
	for did,title_words,body_words,tags in wrap_line_logger(towords_20131030(path),name='wid_counts_20131115'+path):
		for w in tags :
			words[w][0] +=1
		if len(title_words)>0:
			totitle = len(tags)*1.0/len(title_words)
			for w in title_words:
				words[w][1] +=1
				words[w][3] +=totitle
		if len(body_words)>0:
			tobody = len(tags)*1.0/len(body_words)
			for w in body_words:
				words[w][2] +=1
				words[w][4] += tobody
	dic = dictionary_20131030(paths)
	token2id = dic.token2id
	return { token2id.get(w,-1):c for w,c in words.iteritems()  }
@dbcached(db_20131027,'object')
def wid_counts_20131213(paths,path):
	''' { wordid : [#being_tag,#in_title,#in_body,sum(#tags/#words_title),sum(#tags/#words_body) ] }'''
	cnts = wid_counts_20131115(paths,path)
	cnts_all = np.array([ cnt for wid,cnt in cnts.iteritems() ])
	means = [ cnts_all[:,i].mean()  for i in range(5) ]
	vs = [ cnts_all[:,i].var()  for i in range(5) ]
	for wid,cnt in cnts.iteritems():
		for i in range(5):
			cnt[i]=(cnt[i]-means[i])/max(1,vs[i])
	return cnts
	
def load_wid_counts_20131115(options,path):
	hehe = wid_counts_20131115([options.test,options.train],path)
@dbcached(db_20131027,'object')
def wid_coappearnce_20131115(paths,path):
	''' { wordid : [#being_tag,#in_title,#in_body,sum(#tags/#words_title),sum(#tags/#words_title) ] }'''
	words = defaultdict(lambda :[0,0,0,0,0])
	for did,title_words,body_words,tags in wrap_line_logger(towords_20131030(path),name='wid_counts_20131115'+path):
		for w in tags :
			words[w][0] +=1
		if len(title_words)>0:
			totitle = len(tags)*1.0/len(title_words)
			for w in title_words:
				words[w][1] +=1
				words[w][3] +=totitle
		if len(body_words)>0:
			tobody = len(tags)*1.0/len(body_words)
			for w in body_words:
				words[w][2] +=1
				words[w][4] += tobody
	dic = dictionary_20131030(paths)
	token2id = dic.token2id
	return { token2id.get(w,-1):c for w,c in words.iteritems()  }

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

def solution_20131102(options,path,theta=0.05):
	theta = float(theta)
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
				scores = sorted([ (y[i][1],words[i],ys[i]) for i in range(len(y)) if y[i][1]>=theta ],reverse=True)
				yield did,scores #  [ '%s:%s:%s'%(w,y,yy) for y,w,yy in scores ]
				cnt +=1
				if cnt >= atmost:
					break
		@submition_to_path(TMP_DIR+'solution_20131102_%s_%s_%s_%s'%(path.split('/')[-1],batch,loss,penalty))
		def write_preds():
			for did,scores in _make_train_preds():
				yield did,[ w for y,w,yy in scores[:5] ]
		#write_preds()	
		@submition_to_path(TMP_DIR+'score.solution_20131102_%s_%s_%s_%s'%(path.split('/')[-1],batch,loss,penalty))
		def write_scores():
			for did,scores in _make_train_preds():
				yield did,[ ','.join(map(str,x)) for x in scores[:20] ]
		write_scores()	
	make_train_preds(1000,options.batch,options.loss,options.penalty)

def extend_20131117_base(options,base,normalize=False):
	''' return a function(doc,words) that return doc,words'''
	dictionary = dictionary_20131030([options.test,options.train])
	token2id = dictionary.token2id
	id2token = { i:t for t,i in token2id.iteritems()} 
	topwords = [
		(topwords_20131120([options.test,options.train],[1],128,10),1),
		(topwords_20131120([options.train],[3],128,10),2),
	]
	def func(doc,words):
		''' doc => [did,[title_words],[body_words],[tags] ]  
			words => [wid,ks,vs,y]
		'''
		already = { wid for wid,ks,vs,y in words }
		extends = defaultdict(lambda : defaultdict(lambda :[0,0]))
		for wid,ks,vs,y in words:
			w = id2token[wid]
			for i,tops_c in enumerate(topwords):
				tops,c = tops_c
				if w not in tops: continue
				ws = [ (token2id[ww],sim) for ww,sim in tops[w][:c] ]
				ws = [ (wwid,sim) for wwid,sim in ws if wwid not in already ]
				for wwid,sim in ws:
					for j in range(len(ks)):
						if ks[j]<options.base:
							k = ks[j]+(i+1)*200
							extends[wwid][k][0] += vs[j]*sim
							extends[wwid][k][1] += 1
		for wid,kvs in extends.iteritems():
			word = [
				wid,
				[ k for k,v in kvs.iteritems() ],
				[ v[0]/(v[1] if normalize else 1) for k,v in kvs.iteritems() ],
				1 if len(doc)>=4 and id2token[wid] in doc[3] else 0,
			]
			words.append(word)	
		return doc,words
	return func

def extend_20131117(options,path=None):
	return extend_20131117_base(options,0,False)
def extend_20131126(options,path=None):
	return extend_20131117_base(options,options.base,True)
def extend_20131208(options,path=None):
	def load_svds():
		'''yield [did,[(wid,svd_pred,[s_tfidf,s,cnt])] ]'''
		tfidfs = totfidf_20131029([options.train,options.test],[1,2],path,[2])
		suffix = 'train' if path == options.train else 'test'
		idmapping = csv.reader(open('../data/svdfeature.extended.idmapping.'+suffix),delimiter=' ')
		preds = open('../data/pred.txt.'+suffix)
		tfidf = tfidfs.next()
		for sp in idmapping:
			did,tot,ori,ext = map(int,sp[:4])
			pred = [ float(preds.next()) for i in range(tot) ]
			ret = []
			while tfidf[0]<did:
				tfidf = tfidfs.next()
			if tfidf[0] == did and len(tfidf[1])==ori :
				tfidf = sorted(tfidf,key=lambda x:x[1])
				ret = [ (tfidf[i][0],preds[i],[]) for i in range(ori) ]
			sss = [ s.split(',') for s in sp[4:] ]
			ret += [ (int(sss[i][0]),pred[i+ori],map(float,sss[i][1:]))  for i in range(ext) ]
			yield [did,ret]
	svds = load_svds()
	pairs = svds.next()
	cnts = wid_counts_20131115([options.test,options.train],options.train)
	dictionary = dictionary_20131030([options.test,options.train])
	token2id = dictionary.token2id
	id2token = { i:t for t,i in token2id.iteritems()} 
	def func(doc,words):
		''' doc => [did,[title_words],[body_words],[tags] ]  
			words => [wid,ks,vs,y]
		'''
		docid = int(doc[0])
		while pairs[0]<docid:
			pairs[0],pairs[1] = svds.next()
		did,svd = pairs
		if did==docid:
			for wid,svd_pred,ss in svd:
				ori = False
				for _wid,ks,vs,y in words:
					if ks[0]==wid:
						ks.append(250)
						vs.append(svd_pred)
						ori = True
				if not ori :
					cnt = cnts.get(wid,cnts[-1])
					words.append((
						wid,
						range(100,100+len(cnt)) + range(250,254),
						cnt + [ svd_pred ] + map(float,ss) ,
						1 if len(doc)>=4 and id2token[wid] in doc[3] else 0,
					))
		return doc,words
	return func
extention_set = {
	'extend_20131117' : extend_20131117,
	'extend_20131126' : extend_20131126,
	'extend_20131208' : extend_20131208,
	'None' : (lambda doc,words:(doc,words)),
}

def feature_20131116(options,fpath):
	''' yield did,[ wid,ks,vs,y ] '''
	extend = extention_set[options.extend](options,fpath) if options.extend!=None else (lambda d,w:(d,w))
	dictionary = dictionary_20131030([options.test,options.train])
	feas = feature_20131102(fpath)
	fea = feas.next()
	cnts = wid_counts_20131115([options.test,options.train],options.train)
	def log_error(**kargs):
		logging.error('sth wrong with:%s'%(kargs))
	for doc in wrap_line_logger(csv_data(fpath),name='feature_20131116'+fpath,interval=100000):
		tags = set([ dictionary.token2id.get(t,-1) for t in doc[3].split()]) if len(doc)>=4 else []
		while int(fea[0]) < int(doc[0]):
			fea = feas.next()
		words = []
		if fea[0]==doc[0] and len(fea[1])>0:
			for wid,f in fea[1].iteritems():
				y = 1 if wid in tags else 0
				cnt = cnts.get(wid,cnts[-1])
				ks = range(len(f)) + range(100,100+len(cnt))
				vs = f + cnt
				words.append((wid,ks,vs,y))
		else :
			log_error(msg='doc id not match',doc=doc,feature=fea)
		doc,words = extend(doc,words)
		if options.wordid :
			for wid,ks,vs,y in words:
				ks.append(wid+options.base)
				vs.append(1)
		yield doc[0],words

def feature_20131213(options,fpath):
	''' yield did,[wid],[x],[y],[tag] '''
	dictionary = dictionary_20131030([options.test,options.train])
	feas = feature_20131102(fpath)  # [tfidf]*3
	fea = feas.next()
	cnts = wid_counts_20131213([options.test,options.train],options.train) # [#being_tag,#in_title,#in_body,sum(#tags/#words_title),sum(#tags/#words_body) ]
	cotags = [ load_cotags(options,'../data/'+p) for p in ['cotags.1.txt','cotags.2.txt'] ]  # { wid:[(wid,score)] }
	for ct in cotags : 
		for wid,words in ct.iteritems():
			s = sum([ score for cwid,score in words ])
			for i in range(len(words)):
				words[i]=(words[i][0],words[i][1]/s)
	logging.warn('dicts ready!')
	def log_error(**kargs):
		logging.error('sth wrong with:%s'%(kargs))
	for doc in wrap_line_logger(csv_data(fpath),name='feature_20131213'+fpath,interval=100000):
		tags = set([ dictionary.token2id.get(t,-1) for t in doc[3].split()]) if len(doc)>=4 else []
		while int(fea[0]) < int(doc[0]):
			fea = feas.next()
		wids,xs,ys = [],[],[]
		if fea[0]==doc[0] and len(fea[1])>0:
			top_title = sorted([ (wid,f[1]) for wid,f in fea[1].iteritems() ],reverse=True,key=lambda x:x[1])[:3]
			top_body  = sorted([ (wid,f[2]) for wid,f in fea[1].iteritems() ],reverse=True,key=lambda x:x[1])[:10]
			extends = []
			for wid,f in top_title :
				extends += [ cwid for cwid,score in cotags[0][wid][:5] ]
			for wid,f in top_body :
				extends += [ cwid for cwid,score in cotags[1][wid][:3] ]
			def push_out(words):
				other = { wid:[0]*len(x) for wid,x in words.iteritems() }
				for wid,value in words.iteritems():
					for ct in cotags:
						for cwid,s in ct[wid]:
							if cwid in other:
								ovalue = other[cwid]
								other[cwid] = [ ovalue[i]+s*value[i] for i in range(len(ovalue)) ]
				return { wid:words[wid]+other[wid] for wid in words}	
			extends += fea[1].keys()
			words = { wid : fea[1].get(wid,[0,0,0])+cnts.get(wid,cnts[-1]) for wid in set(extends) }
			words = push_out(words)
			# transform basic x to x**2,x**3,x*x' ...
			for wid,f_base in words.iteritems():
				wids.append(wid)
				ys.append( 1 if wid in tags else 0 )
				f_base = [ [x,x**2,x**3,math.exp(x) ] for x in f_base ]
				f_final = []
				for i,x in enumerate(f_base):
					f_final += x
					if options.transform:
						for j in range(i+1,len(f_base)):
							x_other = f_base[j]
							f_final += [ x[k]*x_other[k] for k in range(len(x)) ]
							f_final += [ x[k]/max(1,x_other[k]) for k in range(len(x)) ]
							f_final += [ x_other[k]/max(1,x[k]) for k in range(len(x)) ]
				xs.append(f_final)
		else :
			log_error(msg='doc id not match',doc=doc,feature=fea)
		yield doc[0],wids,xs,ys,tags

def feature_20131215(options,fpath):
	''' yield did,[wid],[x],[y],[tag] '''
	dictionary = dictionary_20131030([options.test,options.train])
	feas = feature_20131102(fpath)  # [tfidf]*3
	fea = feas.next()
	cnts = wid_counts_20131213([options.test,options.train],options.train) # [#being_tag,#in_title,#in_body,sum(#tags/#words_title),sum(#tags/#words_body) ]
	cotags = [ load_cotags(options,'../data/cotags.1.txt',2),load_cotags(options,'../data/cotags.2.txt',3)]
	for ct in cotags:
		for wid,words in ct.iteritems():
			s = sum([ score for cwid,score in words ])
			for i in range(len(words)):
				words[i]=(words[i][0],words[i][1]/s)
	logging.warn('dicts ready!')
	def log_error(**kargs):
		logging.error('sth wrong with:%s'%(kargs))
	for doc in wrap_line_logger(csv_data(fpath),name='feature_20131213'+fpath,interval=100000):
		tags = set([ dictionary.token2id.get(t,-1) for t in doc[3].split()]) if len(doc)>=4 else []
		while int(fea[0]) < int(doc[0]):
			fea = feas.next()
		wids,xs,ys = [],[],[]
		if fea[0]==doc[0] and len(fea[1])>0:
			top_title = sorted([ (wid,f[1]) for wid,f in fea[1].iteritems() ],reverse=True,key=lambda x:x[1])[:2]
			top_body  = sorted([ (wid,f[2]) for wid,f in fea[1].iteritems() ],reverse=True,key=lambda x:x[1])[:4]
			extends = []
			for wid,f in top_title :
				extends += [ cwid for cwid,score in cotags[0][wid] ]
			for wid,f in top_body :
				extends += [ cwid for cwid,score in cotags[1][wid] ]
			def push_out(words):
				other = { wid:[0]*len(x) for wid,x in words.iteritems() }
				for wid,value in words.iteritems():
					for ct in cotags:
						for cwid,s in ct[wid]:
							if cwid in other:
								ovalue = other[cwid]
								other[cwid] = [ ovalue[i]+s*value[i] for i in range(len(ovalue)) ]
				return { wid:words[wid]+other[wid] for wid in words}	
			extends += fea[1].keys()
			words = { wid : fea[1].get(wid,[0,0,0])+cnts.get(wid,cnts[-1]) for wid in set(extends) }
			words = push_out(words)
			for wid,f_base in words.iteritems():
				wids.append(wid)
				ys.append( 1 if wid in tags else 0 )
				f_final = []
				for i,x in enumerate(f_base):
					f_final += [ x,x*x ]
				if options.crossing==1 :
					for i,x in enumerate(f_base):
						for j in range(i+1,len(f_base)):
							y = f_base[j]
							f_final += [ x*y,(x/y if y!=0 else 0),(y/x if x!=0 else 0)]
				xs.append(f_final)
		else :
			log_error(msg='doc id not match',doc=doc,feature=fea)
		yield doc[0],wids,xs,ys,tags

feature_set = {
	'feature_20131116':feature_20131116,
	'feature_20131213':feature_20131213,
	'feature_20131215':feature_20131215,
}

def solution_20131116(options,path,batch,minScore,atmost):
	make_feature = feature_set[options.feature]
	batch = int(batch)
	minScore = float(minScore)
	atmost = int(atmost)
	@dbcached(db_20131027,'object')
	def solution_20131116_model_at_batch(configs,batch):
		raise NotImplementedError('model data is not ready yet')
	def make_matrix(words,dim):
		''' return y,x '''
		idx = [[],[]]
		values = []
		ys = []
		for i,word in enumerate(words):
			wid,ks,vs,y = word
			for ki,key in enumerate(ks):
				idx[0].append(i)
				idx[1].append(key)
				values.append(vs[ki])
			ys.append(y)
		return ys,scipy.sparse.csr_matrix((values,idx),shape=[len(words),dim])
	def make_configs():
		ret = {
			'feature':options.feature,
			'extend':options.extend,
			'base':options.base,
			'batch_size':options.batch_size,
		}
		if options.class_weight!=None :
			ret['class_weight'] = options.class_weight
		if options.alpha!=None and options.alpha!=0.0001 :
			ret['alpha'] = options.alpha
		if options.wordid != 1 : 
			ret['wordid'] = options.wordid
		return ret
	def make_models():
		model = linear_model.SGDClassifier(loss='log',penalty='l1',class_weight=options.class_weight,alpha=options.alpha)
		model.classes_ = np.array([0,1])
		x,ys = [],[]
		@dbcached(db_20131027,'object')
		def solution_20131116_model_at_batch(configs,batch):
			model.partial_fit(x,ys)
			return model
		configs = make_configs()
		feature = make_feature(options,options.train)
		dic = dictionary_20131030([options.test,options.train]).token2id
		dim = len(dic)*options.wordid + options.base
		for batch,values in wrap_line_logger(itertools.groupby(enumerate(feature),lambda x:x[0]/options.batch_size),name='making_models',interval=1):
			words = []
			for i,ws in values:
				did,wds = ws
				words += wds
			ys,x = make_matrix(words,dim)
			solution_20131116_model_at_batch(configs,batch)
	def make_scores():
		'''yield did,[(wid,score,label)]'''
		configs = make_configs()
		dic = dictionary_20131030([options.test,options.train]).token2id
		dic = { wid:w for w,wid in dic.iteritems() }
		dim = len(dic)*options.wordid + options.base
		try :
			model = solution_20131116_model_at_batch(configs,batch)
		except Exception,e:
			traceback.print_stack()
			logging.error(e)
			logging.error('load model failed! trying to train it')
			make_models()
			logging.error('train model done')
			model = solution_20131116_model_at_batch(configs,batch)
		logging.error('model loaded!!')
		for did,words in wrap_line_logger(make_feature(options,path),name='making_scores'):
			if len(words)>0:
				y,x = make_matrix(words,dim)
				probs = model.predict_proba(x)
				score = sorted([ (dic[words[i][0]],probs[i][1],words[i][3],[ (words[i][1][j],words[i][2][j]) for j in range(len(words[i][1])) if words[i][1][j]<options.base ]) for i in range(len(words)) ],key=lambda x:x[1],reverse=True)
			else : score = []
			yield did,score
	@submition_to_path(TMP_DIR+'solution_20131116_%s_%s_%s_%s'%(path.split('/')[-1],batch,minScore,make_configs()))
	def write_preds():
		cnt = 0
		for did,score in make_scores():
			yield did,[ word for word,score,label,feas in score[:10] if score>=minScore  ]
			cnt = cnt +1
			if cnt>=atmost:
				break
	@submition_to_path(TMP_DIR+'score.solution_20131116_%s_%s_%s_%s'%(path.split('/')[-1],batch,minScore,make_configs()))
	def write_scores():
		cnt = 0
		for did,score in make_scores():
			yield did,[ '%s,%s,%s,%s'%(word,score,label,'+'.join([ '%d:%s'%(k,v) for k,v in feas])) for word,score,label,feas in score[:20] if score>=minScore  ]
			cnt = cnt +1
			if cnt>=atmost:
				break
	write_scores()

def solution_20131213(options,path,batch,minScore,atmost):
	make_feature = feature_set[options.feature]
	batch = int(batch)
	minScore = float(minScore)
	atmost = int(atmost)
	@dbcached(db_20131027,'object')
	def solution_20131213_model_at_batch(configs,batch):
		raise NotImplementedError('model data is not ready yet')
	def make_configs():
		ret = {
			'feature':options.feature,
			'extend':options.extend,
			'base':options.base,
			'batch_size':options.batch_size,
		}
		if options.class_weight!=None :
			ret['class_weight'] = options.class_weight
		if options.alpha!=None and options.alpha!=0.0001 :
			ret['alpha'] = options.alpha
		if options.transform!=None and options.transform!=1 :
			ret['transform'] = options.transform
		if options.crossing!=None and options.crossing !=1 :
			ret['crossing'] = options.crossing
		return ret
	def make_models():
		model = linear_model.SGDClassifier(loss='log',penalty='l1',class_weight=options.class_weight,alpha=options.alpha)
		model.classes_ = np.array([0,1])
		xs,ys = [],[]
		@dbcached(db_20131027,'object')
		def solution_20131213_model_at_batch(configs,batch):
			model.partial_fit(xs,ys)
			return model
		configs = make_configs()
		feature = make_feature(options,options.train)
		dic = dictionary_20131030([options.test,options.train]).token2id
		dim = len(dic)*options.wordid + options.base
		for n_batch,values in wrap_line_logger(itertools.groupby(enumerate(feature),lambda x:x[0]/options.batch_size),name='making_models',interval=1):
			xs,ys = [],[]
			for i,ws in values:
				did,wids,x,y,tags = ws
				xs += x
				ys += y
			solution_20131213_model_at_batch(configs,n_batch)
			if n_batch >= batch:
				return
	def make_scores():
		'''yield did,[(word,score,label)]'''
		configs = make_configs()
		dic = dictionary_20131030([options.test,options.train]).token2id
		dic = { wid:w for w,wid in dic.iteritems() }
		try :
			model = solution_20131213_model_at_batch(configs,batch)
		except Exception,e:
			traceback.print_stack()
			logging.error(e)
			logging.error('load model failed! trying to train it')
			make_models()
			logging.error('train model done')
			model = solution_20131213_model_at_batch(configs,batch)
		logging.error('model loaded!!')
		for did,wids,x,y,tags in wrap_line_logger(make_feature(options,path),name='making_scores'):
			if len(wids)>0:
				probs = model.predict_proba(x)
				score = sorted([ (dic[wids[i]],probs[i][1],y[i]) for i in range(len(wids)) ] ,key=lambda x:x[1],reverse=True)
			else : score = []
			yield did,score
	@submition_to_path(TMP_DIR+'solution_20131213_%s_%s_%s_%s'%(path.split('/')[-1],batch,minScore,make_configs()))
	def write_preds():
		cnt = 0
		for did,score in make_scores():
			yield did,[ word for word,score,label in score[:10] if score>=minScore  ]
			cnt = cnt +1
			if cnt>=atmost:
				break
	@submition_to_path(TMP_DIR+'scores.solution_20131213_%s_%s_%s_%s'%(path.split('/')[-1],batch,minScore,make_configs()))
	def write_scores():
		cnt = 0
		for did,score in make_scores():
			yield did,[ '%s:%s:%s'%(word,score,label) for word,score,label in score[:10] if score>=minScore  ]
			cnt = cnt +1
			if cnt>=atmost:
				break
	#write_preds()
	write_scores()

def make_small_feature_set(options,path,fpath,lines):
	''' make lines of <docid,tags,word,wordid,isTag,x...> '''
	lines = int(lines)
	make_feature = feature_set[options.feature]
	dic = dictionary_20131030([options.test,options.train]).token2id
	id2token = { wid:w for w,wid in dic.iteritems() }
	with open(fpath,'w') as fout:
		for did,wids,x,y,tags in wrap_line_logger(make_feature(options,path),name='making_scores'):
			for i in range(len(wids)):
				wid = wids[i]
				fout.write(','.join(map(str,[did]+list(tags)+[id2token[wid],wid,y[i]]+x[i]))+'\n')
			lines = lines -1
			if lines <=0: break
def load_small_feature_set(fpath,max_doc):
	data = [] # (did,wids,xs,ys,tags,words) 
	with open(fpath) as fin:
		reader = csv.reader(fin)
		cnt = 0
		for did,lines in itertools.groupby(reader,lambda x:x[0]):
			wids,xs,ys,tags,words = [],[],[],[],[]
			for sp in lines :
				tags = sp[1:-1507]
				word,wordid,isTag = sp[-1507:-1504]
				wids.append(wordid)
				x = map(float,sp[-1504:])
				xs.append(x)
				ys.append(int(isTag))
				words.append(word)
			data.append((did,wids,xs,ys,tags,words))
			cnt = cnt + 1
			if cnt >= max_doc:
				break
	return data

def try_to_fit(models,data,ntrain,ntest,topn,minq,op=sum):
	xs,ys = [],[]
	for i in range(ntrain[0],ntrain[1]):
		did,wids,x,y,tags,words = data[i]
		xs+=x
		ys+=y
	if len(xs)>0:
		for model in models:
			model.fit(xs,ys)
	fscore,cnt = 0,0
	for i in range(ntest[0],ntest[1]):
		did,wids,x,y,tags,words = data[i]
		ps = [ model.predict_proba(x) for model in models ]
		p = [ op([ ps[j][i][1] for j in range(len(models)) ])  for i in range(len(wids)) ]
		score = sorted([ (p[i],wids[i]) for i in range(len(wids)) if p[i] >=minq ])[:topn]
		target = 1.0*sum([ 1 for s,wid in score if wid in tags ])
		precision = target/max(1,len(score))
		recall = target/max(1,len(tags))
		fscore = fscore + 2*precision*recall/max(1,(precision+recall))
		cnt = cnt +1
	return fscore/max(1,cnt)		

	

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
			if len(line[1].split())==0:
				line[1] = s[1]
			yield line[0],[line[1]]
	_solve()

def judge(options,path,top=5,atmost=1000*10000):
	top,atmost = int(top),int(atmost)
	cnt = 0
	score = 0
	std = csv.reader(open(options.std))
	std.next()
	with open(path) as fin:
		reader = csv.reader(fin)
		reader.next()
		i =0 
		for key,tags in wrap_line_logger(reader,name="judge_of_"+path):
			tags = tags.split()[:top]
			if options.score == 1:
				tags = [ tag.split(':')[0] for tag in tags]
			cnt +=1
			std_tags = []
			for k,tgs in std:
				if k == key :
					std_tags = tgs.split()
					break
			if len(std_tags) > 0:
				common = len([t for t in std_tags if t in tags ])*1.0
				if common >0 :
					p = common / len(tags)
					r = common /len(std_tags)
					score += 2*p*r/(p+r)
			i = i+1
			if i >= atmost : 
				break
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

def small_datas(atmost=1000):
	iters = [
		csv_data('../data/Train.csv'),
		towords_20131030('../data/Train.csv'),
		towordid_20131029('../data/Train.csv',[2]),
		totfidf_20131029(['../data/Train.csv','../data/Test.csv'],[1,2],'../data/Train.csv',[2])
	]
	objs = [
	
	]
	return [ [ it.next() for i in range(atmost) ] for it in iters ] + objs

def to_svd_feature(options,outpath):
	with open(outpath,'w') as fout:
		for path in [options.test,options.train]:
			tfidfs = wrap_line_logger(totfidf_20131029([options.train,options.test],[1,2],path,[2]),name='to_svd_feature:%s'%(path))
			for did,tfidf in tfidfs:
				tfidf = sorted(tfidf,key=lambda x:x[1])
				nu = len(tfidf)
				fu = ' '.join([ '%d:%.3f'%(k,v) for k,v in tfidf])
				targets = tfidf[:3]+ tfidf[nu/2-1:nu/2+2] +tfidf[-3:] if nu>10 else tfidf
				for k,v in targets:
					fout.write('%.3f 1 %d 1 1:1 %s %d:1\n'%(v,nu,fu,k))

def to_svd_feature_group(options,outpath,cotags=''):
	cotags = load_cotags(options,cotags) if len(cotags)>0 else defaultdict(list)
	with open(outpath,'w') as fout:
	 with open(outpath+'.group','w') as gout:
	  with open(outpath+'.idmapping','w') as mout:
		for path in [options.test,options.train]:
			tfidfs = wrap_line_logger(totfidf_20131029([options.train,options.test],[1,2],path,[2]),name='to_svd_feature:%s'%(path))
			for did,tfidf in tfidfs:
				tfidf = sorted(tfidf,key=lambda x:x[1])
				extends = defaultdict(lambda :(0,0,0))
				for k,v in tfidf[:10]:
					for wid,score in cotags[k]:
						s_tfidf,s,cnt = extends[wid]
						extends[wid] = (s_tfidf+v,s+score,cnt+1)
				extends = sorted(extends.items(),key=lambda x:x[1],reverse=True)[:30]
				nu = len(tfidf)
				fu = ' '.join([ '%d:%.3f'%(k+10,v) for k,v in tfidf])
				targets = tfidf + [ (k,0) for k,ss in extends ]
				for k,v in targets:
					fout.write('%.3f 0 0 1 %d:1\n'%(v,k+10))
				gout.write('%d %d %s\n'%(len(targets),len(tfidf),fu))
				mout.write('%s %d %d %d %s\n'%(did,len(targets),len(tfidf),len(extends),' '.join([ str(k)+','+','.join(map(str,ss)) for k,ss in extends]) ))

def tag_svd_feature_group(options,path,outpath):
		'''yield [did,[(wid,svd_pred,[s_tfidf,s,cnt])] ]'''
		tfidfs = totfidf_20131029([options.train,options.test],[1,2],path,[2])
		suffix = 'train' if path == options.train else 'test'
		idmapping = csv.reader(open('../data/svdfeature.extended.idmapping.'+suffix),delimiter=' ')
		preds = open('../data/pred.txt.'+suffix)
		tfidf = tfidfs.next()
		for sp in idmapping:
			did,tot,ori,ext = map(int,sp[:4])
			pred = [ float(preds.next()) for i in range(tot) ]
			ret = []
			while tfidf[0]<did:
				tfidf = tfidfs.next()
			if tfidf[0] == did and len(tfidf[1])==ori :
				tfidf = sorted(tfidf,key=lambda x:x[1])
				ret = [ (tfidf[i][0],preds[i],[]) for i in range(ori) ]
			sss = [ s.split(',') for s in sp[4:] ]
			ret += [ (int(sss[i][0]),pred[i+ori],map(float,sss[i][1:]))  for i in range(ext) ]
			yield [did,ret]


def same_title(options):
	def make_input():
		for line in wrap_line_logger(sys.stdin,name='sample_title.lines'):
			sp = line[:-1].split('\t')
			if len(sp)>=2:
				yield sp
	for title,its in wrap_line_logger(itertools.groupby(make_input(),lambda x:x[1]),name='same_title.groups'):
		dids,tags = [],[]
		for sp in its:
			dids.append(sp[0])
			tags.append(sp[2] if len(sp)>=3 else "")
		if len(dids)>=2:
			print title+'\t'+'\t'.join([','.join(dids),','.join(tags)])
@dbcached(db_20131027,'object')
def tag_of_same_title(path):
	ans = {}
	with open(path) as fin:
		for line in wrap_line_logger(fin,name='tag_of_same_title:'+path):
			sp = line[:-1].split('\t')
			tags = []
			for tgs in sp[2].split(','):
				tags += tgs.split()
			tags = list(set(tags))
			for did in sp[1].split(','):
				ans[int(did)] = tags
	return ans

def correct_by_same_title(options,path,min_s=0.0):
	@submition_to_path(path+'.currected.%s.txt'%min_s)
	def run():
		his = tag_of_same_title('../data/same_title')
		reader=csv.reader(open(path))
		reader.next()
		for did,words in wrap_line_logger(reader,name='correcting:'+path):
			old = his.get(int(did),[])
			if len(old)>0 :
				yield did,old
			else :
				words = words.split()
				if options.score == 0 :
					yield did,words
				else :
					words = [ word.split(':')  for word in words ]
					yield did,[ word for word,score,y in words if float(score)>=min_s ]
	run()

def make_wid_pairs(options,paths,row):
	paths = paths.split(',')
	row = [int(row)]
	for path in paths:
		wordids = towordid_20131029(path,row)
		for did,words in wordids:
			for i in range(len(words)):
				w1,c1 = words[i]
				for j in range(i+1,len(words)):
					w2,c2 = words[j]
					print w1,c1,w2,c2
					print w2,c2,w1,c1

def count_cowords_reduce(options):
	''' <w1,c1,w2,c2> => <w1,sum(c1),(wi,sum(bool),sum(c1),sum(c2),sum(c1*c2)) )>'''
	reader = csv.reader(sys.stdin,delimiter=' ')
	for wid,its in wrap_line_logger(itertools.groupby(reader,lambda x:x[0]),name='count_cowords_reduce',interval=1000):
		sums = defaultdict(lambda :(0,0,0,0))
		sw1 = 0
		for w1,c1,w2,c2 in its:
			c1,w2,c2=int(c2),int(w2),int(c2)
			sw1 += c1
			ss = sums[w2]
			sums[w2] = (
				ss[0] + 1,
				ss[1] + c1,
				ss[2] + c2,
				ss[3] + c1*c2,
			)
		top10 = sorted(sums.items(),key=lambda x:x[1],reverse=True)[:10]
		print ','.join([wid,str(sw1)]+[ str(w2)+' '+' '.join(map(str,ss)) for w2,ss in top10 ])	

def cotags_map(options,row):
	row = [int(row)]
	wordids = totfidf_20131029([options.train,options.test],[1,2],options.train,row)
	raw = csv_data(options.train)
	token2id = dictionary_20131030([options.test,options.train]).token2id
	for did,words in wrap_line_logger(wordids,name='cotags_map'):
		sp = raw.next()
		while int(sp[0])!=int(did):
			sp = raw.next()
		tags = sp[3]
		for wid,cnt in words :
			print wid,cnt,' '.join(map(str,[ token2id[t] for t in tags.split()]))

def cotags_reduce(options):
	reader = csv.reader(sys.stdin,delimiter=' ')
	for wid,its in wrap_line_logger(itertools.groupby(reader,lambda x:x[0]),name='cotag_reduce',interval=1000):
		tags = defaultdict(int)
		tall = 0
		for sp in its:
			wd,tfidf=sp[:2]
			tfidf = float(tfidf)
			for tag in sp[2:]:
				tags[tag] += tfidf
			tall += tfidf
		top10 = sorted(tags.items(),reverse=True,key=lambda x:x[1])[:10]
		print ','.join([wid,str(tall)]+[ '%s %s'%(tag,w) for tag,w in top10 ])

def load_cotags(options,path,topn=100000):
	'''return { wid : [(wid,%tfidf)]  }'''
	ret = defaultdict(list)
	with open(path) as fin:
		reader = csv.reader(fin)
		for sp in reader:
			wid,tall = int(sp[0]),float(sp[1])
			ret[wid] = [ (int(ctid),float(tf)/tall) for ctid,tf in [ x.split() for x in sp[2:topn] ] ]
	return ret

def split_svd(options,grp,prd):
	''' split file into train & test part '''
	with open(grp) as gin:
	 with open(prd) as pin:
	  with open(grp+'.train','w') as gout_train:
	   with open(grp+'.test','w') as gout_test:
	    with open(prd+'.train','w') as prd_train:
		 with open(prd+'.test','w') as prd_test:
			reader = csv.reader(gin,delimiter=' ')
			for sp in reader:
				wid,tot = sp[:2]
				if int(wid)<6034196:
					pout,gout = prd_train,gout_train
				else :
					pout,gout = prd_test,gout_test
				gout.write(' '.join(sp)+'\n')
				for i in range(int(tot)):
					p = pin.next()
					pout.write(p)

def call_without_options(options,*args):
	globals()[args[0]](*(args[1:]))

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
	parser.add_option('', '--batch_size', dest='batch_size',type=int,help='model batch size',default=40000);
	parser.add_option('', '--feature', dest='feature',type=str,help='using feature');
	parser.add_option('', '--extend', dest='extend',type=str,help='word extend');
	parser.add_option('', '--base', dest='base',type=int,help='base feature slot len',default=1000);
	parser.add_option('', '--class_weight', dest='class_weight',type=str,help='class weight',default=None);
	parser.add_option('', '--alpha', dest='alpha',type=float,help='alpha of regulization',default=0.0001);
	parser.add_option('', '--wordid', dest='wordid',type=int,help='weather to use wordid in feature',default=1);
	parser.add_option('', '--transform', dest='transform',type=int,help='whether to transform feature from x=>x*x',default=1);
	parser.add_option('', '--validation', dest='validation',type=int,help='whether to log out training fscore',default=0);
	parser.add_option('', '--score', dest='score',type=int,help='whether the input submition is make up of score',default=0);
	parser.add_option('', '--crossing', dest='crossing',type=int,help='whether the feature should have crossing multiplication',default=1);
	
	global options
	(options, args) = parser.parse_args()
	logging.warn((options,args))
	random.seed(options.seed)
	try:
		globals()[options.job](options,*args)
	except Exception,e:
		traceback.print_exc()
		logging.error(e)
		raise e

if __name__ == '__main__':
	main()
