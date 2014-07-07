import os,cPickle,logging,pickle

def write_to_file(path,seperator=' '):
	def wrapper(func):
		def _func(*args,**karg):
			with open(path,'w') as fout:
				for sp in func(*args,**karg):
					line = seperator.join(sp)
					fout.write(line)
					fout.write('\n')
		return _func
	return wrapper

def try_get_path(path):
	dirname,filename = os.path.split(path)
	if len(filename) <= 220 :
		return path
	db_path = 'path_mapping.cPickle'
	if os.path.exists(db_path) :
		with open(db_path) as fin :
			db = cPickle.load(fin)
	else :
		db = {}
	if filename not in db :
		db[filename] = str(len(db))+'.cPickle'
		with open(db_path,'w') as fout :
			pickle.dump(db,fout) 
	filename =  db[filename]
	return os.path.join(dirname,filename)

def disk_cached(prefix):
	def wrapper(func):
		def _func(*args,**karg):
			value = None
			name = '%s-%s'%(args,karg)
			name = name.replace('/','__')
			path = '%s%s.cPickle'%(prefix,name)
			path = try_get_path(path)
			if os.path.exists(path):
				logging.info('pickle exists : %s'%(path))
				with open(path) as fin:
					value = pickle.load(fin)
				logging.info('pickle loaded : %s'%(path))
			else :
				logging.info('pickle not exists : %s'%(path))
				value = func(*args,**karg)
				with open(path,'w') as fout:
					pickle.dump(value,fout)
				logging.info('pickle written : %s'%(path))
			return value
		return _func
	return wrapper

def memory_cached(func):
	cache = {}
	list_cache = []
	def _func(*args):
		value = None
		try : 
			if args in cache:
				value = cache[args]
			else :
				value = func(*args)
				cache[args] = value
		except :
			for k,v in list_cache :
				if k==args:
					value = v
					break
			if not value :
				value = func(*args)
				list_cache.append((args,value))
		return value
	return _func
