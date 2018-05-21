import warnings
warnings.filterwarnings('ignore')
from contextlib import contextmanager
import time
import pickle

root = '/kaggle/competitions/avito-demand-prediction/'

@contextmanager
def timeit(label):
    print('{} started'.format(label))
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print('{}: {}'.format(label, end - start))

def dump_obj(obj, name):
    with timeit('dump_obj ' + name):
        pickle.dump(obj, open(root + 'features/{}.pkl'.format(name), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)    

def load_obj(name):
    with timeit('load_obj ' + name):
        return pickle.load(open(root + 'features/{}.pkl'.format(name), 'rb'))