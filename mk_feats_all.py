# coding: utf-8
"""
extract features
"""
import numpy as np
import pandas as pd

import copy
import nltk
import re
import pickle
import warnings
import gc

from string import punctuation

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from scipy.cluster.hierarchy import fcluster, linkage

from gensim.models import Word2Vec
from scipy.sparse import hstack, csr_matrix
from random import shuffle

warnings.filterwarnings('ignore')

root = '/kaggle/competitions/avito-demand-prediction/'

def dump_matrix(matrix, name):
    pickle.dump(matrix, open(root + 'features/{}.pkl'.format(name), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)    

def load_matrix(name):
    return pickle.load(open(root + 'features/{}.pkl'.format(name), 'rb'))

print('Loading data')

tr = pd.read_csv(root+'train.csv.zip', parse_dates=['activation_date'])  # 1503424
te = pd.read_csv(root+'test.csv.zip',  parse_dates=['activation_date'])  # 508438

tr.sort_values('activation_date', inplace=True)
te.sort_values('activation_date', inplace=True)

print('Max train date:', tr.activation_date.max())
print('Min test date:', te.activation_date.min())
print('Data shape is', tr.shape, te.shape)
gp = pd.read_csv(root+'aggregated_features.csv')
#tr = tr.merge(gp, on='user_id', how='left')
#te = te.merge(gp, on='user_id', how='left')
y = tr.deal_probability
tr_index = tr.index
te_index = te.index
tr_item_id = tr['item_id']
te_item_id = te['item_id']
print(len(tr_index),len(te_index))
dump_matrix(tr_index,'tr_index')
dump_matrix(te_index,'te_index')
dump_matrix(tr_item_id,'tr_item_id')
dump_matrix(te_item_id,'te_item_id')
dump_matrix(y, 'y')

daset = pd.concat([tr, te], axis=0)

print('Daset shape rows:{} cols:{}'.format(*daset.shape))
del tr
del te
gc.collect()

cat_cols = ['region', 'parent_category_name', 'user_type', 'user_id']
num_cols = ['price', 'image_top_1', 'item_seq_number']
txt_cols = ['city', 'category_name', 'param_1', 'title', 'description']

num_cols += list(gp.columns)[1:]

dump_matrix(daset[list(gp.columns)[1:]],'daset_aggregated_features.pkl')

daset['n_missing_features'] = daset.drop('deal_probability', axis=1).isnull().sum(axis=1)
num_cols.append('n_missing_features')

del daset['image']
gc.collect()

print(50 * '/')
print('=====activation_date value coutns======')
print(daset.activation_date.value_counts().sort_index())
print(50 * '/')


print('Preprocessing .... ')

daset['day_of_week'] = daset.activation_date.dt.weekday
num_cols.append('day_of_week')

print('Base categorical feature')
for c in ['region', 'parent_category_name', 'user_type', 'day_of_week']:
    d = daset[c].value_counts().to_dict()
    daset[c + '_factor'] = daset[c].apply(lambda x: d.get(x, 0)).astype(int)
    num_cols.append(c + '_factor')

print('Handle missing')
daset["price"] = np.log1p(daset["price"]) + 0.001
daset["price"].fillna(-1, inplace=True)
daset["image_top_1"] = daset["image_top_1"].fillna(-1).astype(int)
daset["item_seq_number"] = daset["item_seq_number"].astype(int)


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(trn_series=None,
                  tst_series=None,
                  target=None,
                  min_samples_leaf=1,
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)


print('target_encode ...')
target_enc_cols = [
    'region', 'image_top_1', 'category_name', 'user_type',
    'parent_category_name','day_of_week'
]
for c in target_enc_cols:
    print(daset.loc[tr_index, c].shape,y.shape)
    tr_avg, te_avg = target_encode(daset.loc[tr_index, c].fillna('nan'),
                                   daset.loc[te_index, c].fillna('nan'), y)
    cc = 'n_' + c + '_target_avg'
    daset.loc[tr_index, cc] = tr_avg.values
    daset.loc[te_index, cc] = te_avg.values
    num_cols.append(cc)

print('Add uv')
for c in ['region', 'parent_category_name', 'user_type', 'city','day_of_week',
          'category_name', 'image_top_1', 'item_seq_number']:
    pv = daset[c].value_counts()
    uv = daset.user_id.groupby(daset[c]).agg(lambda x: len(np.unique(x)))
    if len(pv) > 1000:
        pv = pv[:1000]
        uv = uv[:1000]

    upv = pv / uv
    daset[c + '_uv'] = daset[c].map(uv).fillna(-1).astype(int)
    daset[c + '_upv'] = daset[c].map(upv).fillna(-1).astype(float)
    num_cols.append(c + '_uv')
    num_cols.append(c + '_upv')


def apply_w2v(sentences, model, num_features):
    def _average_word_vectors(words, model, vocabulary, num_features):
        feature_vector = np.zeros((num_features,), dtype="float64")
        n_words = 0.
        for word in words:
            if word in vocabulary:
                n_words = n_words + 1.
                feature_vector = np.add(feature_vector, model[word])

        if n_words:
            feature_vector = np.divide(feature_vector, n_words)
        return feature_vector

    vocab = set(model.wv.index2word)
    feats = [_average_word_vectors(s, model, vocab, num_features) for s in sentences]
    return csr_matrix(np.array(feats))


print('add_target_cluster')


def add_target_cluster(c):
    print('add_target_cluster {}'.format(c))
    gkey = c
    _nan = daset.loc[daset[c].isnull()].shape[0] > 0
    if _nan:
        gkey = c + '_tmp'
        daset[gkey] = daset[c].fillna('NAN')
    gp = pd.concat([daset.loc[tr_index, gkey], y], axis=1).groupby(gkey)['deal_probability']
    hist = gp.agg(lambda x: ' '.join(x.apply(lambda i: round(i, 2)).astype(str)))
    gp_index = hist.index
    sentences = [x.split(' ') for x in hist.values]

    n_features = 300
    w2v = Word2Vec(sentences=sentences, min_count=1, size=n_features)
    w2v_feature = apply_w2v(sentences, model=w2v, num_features=n_features)

    del sentences, hist
    gc.collect()

    # clustering
    sims = cosine_similarity(w2v_feature)
    Z = linkage(sims, 'ward')
    max_dist = 1.0
    cluster_labels = fcluster(Z, max_dist, criterion='distance')
    cluster_labels = pd.Series(cluster_labels, name=c + '_cluster', index=gp_index)
    daset[c + '_cluster'] = daset[gkey].map(cluster_labels).fillna(-1).astype(int)
    num_cols.append(c + '_cluster')
    if _nan:
        del daset[gkey]
    del sims, Z, cluster_labels
    gc.collect()


for c in ['image_top_1', 'param_1', 'category_name', 'city', 'day_of_week']:
    add_target_cluster(c)

print('Cat2Vec...')
n_cat2vec_feature = 10
n_cat2vec_window = 5


def gen_cat2vec_sentences(data):
    X_w2v = copy.deepcopy(data)
    names = list(X_w2v.columns.values)
    for c in names:
        X_w2v[c] = X_w2v[c].fillna('nan').astype('category')
        X_w2v[c].cat.categories = ["%s:%s" % (c, g) for g in X_w2v[c].cat.categories]
    X_w2v = X_w2v.values.tolist()
    return X_w2v


def fit_cat2vec_model():
    X_w2v = gen_cat2vec_sentences(daset.loc[:, cat_cols].sample(frac=0.8))
    for i in X_w2v:
        shuffle(i)
    model = Word2Vec(X_w2v, size=n_cat2vec_feature, window=n_cat2vec_window)
    return model

print('fit_cat2vec_model')
c2v_model = fit_cat2vec_model()
print('apply_w2v for cat2vec')
tr_c2v_matrix = apply_w2v(gen_cat2vec_sentences(daset.loc[tr_index, cat_cols]), c2v_model, n_cat2vec_feature)
te_c2v_matrix = apply_w2v(gen_cat2vec_sentences(daset.loc[te_index, cat_cols]), c2v_model, n_cat2vec_feature)
dump_matrix(tr_c2v_matrix, 'tr_c2v_matrix')
dump_matrix(te_c2v_matrix, 'te_c2v_matrix')
del tr_c2v_matrix, te_c2v_matrix


print('Text feature...')
daset['n_char_title'] = daset.title.apply(len)
daset['n_char_description'] = daset.description.fillna('').apply(len)
num_cols.append('n_char_title')
num_cols.append('n_char_description')


def fn_word_count(x):
    return len(x.split(' '))


def cat_params(x):
    params = ' '.join(x)
    params = params.lower().strip()
    return params


print('parmas...')
pms_cols = ['param_1', 'param_2', 'param_3']
daset['params'] = daset[pms_cols].fillna('исключение').apply(cat_params, axis=1)
del daset['param_2'], daset['param_3']
gc.collect()

daset['n_char_params'] = daset.params.apply(len)
daset['n_word_params'] = daset.params.apply(fn_word_count)
num_cols.append('n_char_params')
num_cols.append('n_word_params')

pms_tv = CountVectorizer(
    max_features=10000,
    ngram_range=(1, 6),
    strip_accents='unicode')

print('Params Model CountVectorizer')
pms_tv.fit(daset['params'].sample(frac=0.7))
print('transform params to vector')
tr_param_vec_matrix = pms_tv.transform(daset.loc[tr_index, 'params'])
te_param_vec_matrix = pms_tv.transform(daset.loc[te_index, 'params'])


dump_matrix(pms_tv, 'pms_tv')
dump_matrix(daset['params'], 'daset_params')
dump_matrix(tr_param_vec_matrix, 'tr_param_vec_matrix')
dump_matrix(te_param_vec_matrix, 'te_param_vec_matrix')
del tr_param_vec_matrix, te_param_vec_matrix, pms_tv, daset['params']
gc.collect()

print('text ...')
wpt = nltk.WordPunctTokenizer()
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('russian')
stop_words.extend(nltk.corpus.stopwords.words('english'))

def text_mask(x):
    return x not in set(punctuation) and x not in stop_words

text_rule = re.compile(r'[^\u0400-\u0527\s]')


def cat_text(x):
    text = ' '.join(x)
    text = text.lower()
    text = text_rule.sub('', text)
    text = text.strip()
    return ' '.join(filter(text_mask, wpt.tokenize(text)))

print('Text generating')
daset['text'] = daset[txt_cols].fillna('').apply(cat_text, axis=1)

daset['n_word_text'] = daset['text'].apply(fn_word_count)
num_cols.append('n_word_text')

daset.drop(txt_cols, axis=1, inplace=True)
gc.collect()

txt_tv = CountVectorizer(
    max_features=50000,
    ngram_range=(1, 3),
    stop_words=stop_words,
    analyzer='word')

print('Text Model CountVectorizer')
txt_tv.fit(daset['text'].sample(frac=0.7))
print('transform text to vector')
tr_txt_vec_matrix = txt_tv.transform(daset.loc[tr_index, 'text'])
te_txt_vec_matrix = txt_tv.transform(daset.loc[te_index, 'text'])

dump_matrix(txt_tv, 'txt_tv')
dump_matrix(daset['text'], 'daset_text')
dump_matrix(tr_txt_vec_matrix, 'tr_txt_vec_matrix')
dump_matrix(te_txt_vec_matrix, 'te_txt_vec_matrix')
del tr_txt_vec_matrix, te_txt_vec_matrix, txt_tv, daset['text']
daset.drop(cat_cols, axis=1, inplace=True)
gc.collect()

dump_matrix(daset.loc[:, num_cols], 'daset_num_cols')

feature_names = num_cols \
    + ['c2v_%d' % i for i in range(n_cat2vec_feature)] \
    + ['p2v_%s' % c for c in range(pms_tv.get_feature_names())] \
    + ['t2v_%s' % i for c in range(txt_tv.get_feature_names)]
dump_matrix(feature_names, 'feature_names')

print('Make Feature Completed.')
# daset_num_cols = load_matrix('daset_num_cols')
# tr_c2v_matrix = load_matrix('tr_c2v_matrix')
# tr_param_vec_matrix = load_matrix('tr_param_vec_matrix')
# tr_txt_vec_matrix = load_matrix('tr_txt_vec_matrix')
# tr_list = [daset_num_cols.loc[tr_index].values, tr_c2v_matrix, tr_param_vec_matrix, tr_txt_vec_matrix]
# X = hstack(tr_list).tocsr()
# dump_matrix(X, 'X')
# del tr_list, tr_c2v_matrix,tr_param_vec_matrix,tr_txt_vec_matrix
# gc.collect()


# te_c2v_matrix = load_matrix('te_c2v_matrix')
# te_param_vec_matrix = load_matrix('te_param_vec_matrix')
# te_txt_vec_matrix = load_matrix('te_txt_vec_matrix')
# te_list = [daset_num_cols.loc[te_index].values, te_c2v_matrix, te_param_vec_matrix, te_txt_vec_matrix]
# X_te = hstack(te_list).tocsr()
# dump_matrix(X_te, 'X_te')
# del te_list, te_c2v_matrix, te_param_vec_matrix, te_txt_vec_matrix
# gc.collect()

# feature_dump = 'stage1_features.pkl'
# dataset = {
#     'X': X,
#     'X_te': X_te,
#     'y': y,
#     'tr_index': tr_index,
#     'te_index': te_index,
#     'feature_names': feature_names
# }
# print('Dumpled dataset.', feature_dump)


