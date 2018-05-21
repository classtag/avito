# coding: utf-8
import pandas as pd
from gensim.models import Word2Vec
import gc
import logging
from pymystem3 import Mystem
import string
import pickle
import nltk
import re
import multiprocessing as mp

nltk.download('punkt')

logging.basicConfig(level=logging.INFO)
root = '/kaggle/competitions/avito-demand-prediction/'


def dump_matrix(matrix, name):
    pickle.dump(matrix, open(root + 'features/{}.pkl'.format(name), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)    

def load_matrix(name):
    return pickle.load(open(root + 'features/{}.pkl'.format(name), 'rb'))

stopwords = nltk.corpus.stopwords.words('russian')
punctuation = set(string.punctuation)
emoji = load_matrix('emoji')
chuncksize = 100000
usecols = ['param_1', 'param_2', 'param_3', 'title', 'description']
model = Word2Vec(size=100, window=5, max_vocab_size=500000)
update = False

rule_1 = re.compile("\d+х\d+х\d+")
rule_2 = re.compile("\d+х\d+")
rule_3 = re.compile("\d*[-|–]\d*")
rule_4 = re.compile("\d*\\.\d*")
rule_5 = re.compile("([^\W\d_]+)(\d+)")
rule_6 = re.compile("(\d+)([^\W\d_]+)")
rule_7 = re.compile("\d+\\/\d|\d+-к|\d+к|\\.\/|\d+х\d+х\d+|\d+х\d+")
rule_8 = re.compile("\\s+")
rule_9 = re.compile("([nn\\s]+)")
rule_10 = re.compile("\d+nn")
rule_11 = re.compile("\d+n")
def normalize_text(s):
    s = rule_1.sub('nxnxn ', s)
    s = rule_2.sub('nxn ', s)
    s = rule_3.sub('nn ', s)
    s = rule_4.sub('n ', s)
    s = rule_5.sub(lambda m: 'n' + m.group(1) + ' ', s)
    s = rule_6.sub(lambda m: 'n' + m.group(2) + ' ', s)
    s = rule_7.sub(' ', s.lower())
    
    s = ''.join([c if c.isalpha() or c.isalnum() or c.isspace() else ' ' for c in s if s not in emoji and s not in punctuation and not s.isnumeric()])
    s = rule_8.sub(' ', s)
    s = rule_9.sub('nn ', s)
    s = rule_10.sub('nn ', s)
    s = rule_11.sub('n ', s)
    s = s.strip()
    words = [w for w in s.split(' ') if w not in stopwords]
    return ' '.join(words)

def prepare_text(file):
    sentences_file = root+'features/{}_sentences.txt'.format(file)
    print('prepare_text {}'.format(sentences_file))
    i = 0
    with open(sentences_file, 'w') as sf:
        for df in pd.read_csv(root + file + '.csv.zip', chunksize=10000, usecols=usecols):
            df['text'] = df['param_1'].str.cat([df.param_2, df.param_3, df.title, df.description], sep=' ', na_rep='')
            sentences = df['text'].apply(normalize_text).values
            sf.writelines([s + '\n' for s in sentences])
            i += len(sentences)
            del sentences, df
            gc.collect()
            if i>20000000:
                break;
            print('Write {} {}W text'.format(file, i // 10000))
    print(file+" complated")


def fit_w2v(sentences_file):
    global update
    print(sentences_file)
    with open(sentences_file, 'r') as sf:
        sentences = []
        while True:
            s = sf.readline()
            if s is None:
                break
            sentences.append(s)
            if len(sentences) > 100000:

                sentences = [s.split() for s in sentences]
                model.build_vocab(sentences, update=update)
                model.train(sentences, total_examples=model.corpus_count, epochs=10)
                update = True
                sentences = []

files = ['train', 'test', 'train_active']
#files = ['train_active']
#for file in files:
#    p = mp.Process(target=prepare_text, args=(file,))
#    p.start()
#p.join()
#print('prepare_text completed')


for file in files:
    sentences_file = root+'features/{}_sentences.txt'.format(file)
    for k in range(10):
        print(20 * '=' + 'Epoch {}, File {}'.format(k, file) + 20 * '=')
        fit_w2v(sentences_file)
    print(30 * '=' + '{} train finished'.format(file) + '=' * 30)
    model.save(root+'features/avito.w2v')
    print('Finished')
