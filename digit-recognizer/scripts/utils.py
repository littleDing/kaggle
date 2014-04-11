import os,json,logging,traceback,sqlite3,datetime
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

