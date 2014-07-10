import os,json,logging,traceback,sqlite3,datetime
import pandas as pd
from decorators import memory_cached
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(funcName)s@%(filename)s#%(lineno)d %(levelname)s %(message)s')

PWD = os.path.split(os.path.abspath(__file__))[0]

CONFIGS = {}
try :
	with open(os.path.join(PWD,'../configs.json')) as fin :
		CONFIGS = json.load(fin)
except Exception,e:
	traceback.print_exc()
	logging.error(e)

ROOT_DIR = CONFIGS.get('root_dir',os.path.join(PWD,'../'))
ROOT_DIR = ROOT_DIR if ROOT_DIR else os.path.join(PWD,'../')
def load_dir(suffix):
	path = os.path.join(ROOT_DIR,suffix)
	if not os.path.exists(path):
		os.makedirs(path)
	return path
CACHE_DIR = load_dir('caches')
DATA_DIR = load_dir('data')
ANS_DIR = load_dir('answers')
TEMP_DIR = load_dir('temp')
RGF_TEMP_DIR = load_dir('temp/rgf')

HDFS = CONFIGS.get('hdfs')
HDFS_HOST = 'hdfs://'+HDFS.get('host','localhost:54310') + '/'
HDFS_DATA_DIR 	= HDFS_HOST + HDFS.get('data',None)
HDFS_CACHE_DIR 	= HDFS_HOST + HDFS.get('cache',None)

def get_date(s):
	return datetime.date(*map(int,s.split('-')))

def get_periods(from_date,to_date=None,delta=None):
	if type(from_date)==str :
		from_date = get_date(from_date)
	if type(to_date) == str:
		to_date = get_date(to_date)
	if type(delta) == str:
		delta = int(delta)
	if delta == None:
		delta = (to_date - from_date).days +1
	return [ from_date+datetime.timedelta(i) for i in range(delta) ]

@memory_cached
def read_csv(filename):
	return pd.read_csv(os.path.join(DATA_DIR,filename),true_values='t',false_values='f')

@memory_cached
def spark():
	conf = CONFIGS.get('spark',{})
	from pyspark import SparkContext
	sc = SparkContext(conf.get('host','local'))
	return sc

def to_rdd(filename,sep=',',true_values=None,false_values=None):
	if true_values == None and false_values == None:
		def spliter(s):
			sp = s.split(sep)
			return sp[0],sp[1:]
	else :
		def spliter(s):
			sp = s.split(sep)
			value = map(lambda x:1 if x in true_values else x,sp[1:])
			value = map(lambda x:0 if x in false_values else x,value)
			return sp[0],value
	sc = spark()
	path = os.path.join(HDFS_DATA_DIR,filename)
	return sc.textFile(path).map(spliter)


