
# coding: utf-8

# In[10]:


# coding: utf-8
"""
train lightgbm model
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import lightgbm as lgb

import gc
import pickle
from datetime import datetime
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

file_stamp = datetime.now().strftime('%m%d%H%M')

root = '/kaggle/competitions/avito-demand-prediction/'

def load_matrix(name):
    return pickle.load(open(root + 'features/{}.pkl'.format(name), 'rb'))


# In[2]:


tr_index = load_matrix('tr_index')
te_index = load_matrix('te_index')

X = load_matrix('X_20180519')
y = load_matrix('y')
X_te = load_matrix('X_20180519_te')
feature_names = load_matrix('lgb_feature_20180519')


# In[3]:


def cut_feature(feature_importance_):
    global feature_names
    feature_weak_ = [x[0] for x in feature_importance_ if x[1]==0.0]
    feature_index = [i for i, c in enumerate(feature_names) if c not in feature_weak_]
    _feature_names = [c for i, c in enumerate(feature_names) if c not in feature_weak_]
    _X    = sparse.lil_matrix(X[:,feature_index])
    _X_te = sparse.lil_matrix(X_te[:,feature_index])
    print('cut feature {}%'.format(100*len(feature_weak_)/len(feature_importance_)))
    return _X, _X_te, _feature_names


# In[4]:


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# In[5]:


num_boost_round = 20000
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.1,
    'num_leaves': 15,
    'max_bin': 256,
    'feature_fraction': 0.6,
    'min_child_samples': 10,
    'min_child_weight': 150,
    'min_split_gain': 0,
    'subsample': 0.9,
    'drop_rate': 0.1,
    'max_drop': 50,
    'verbosity': 0,
    'seed': 228,
}
n_folds = 5
kfold = StratifiedKFold(n_splits=n_folds, shuffle=False, random_state=218)


# In[6]:


# feature selection


# In[7]:


val_split_idx = round(X.shape[0]*.9)
X_tr, X_va, y_tr, y_va = X[:val_split_idx], X[val_split_idx:], y[:val_split_idx], y[val_split_idx:]
print(X_tr.shape, X_va.shape, y_tr.shape[0], y_va.shape[0])
dtrain = lgb.Dataset(X_tr, y_tr, feature_name=feature_names)
dvalid = lgb.Dataset(X_va, y_va, feature_name=feature_names, reference=dtrain)

params['learning_rate'] = 0.2
params['feature_fraction'] = 0.66
gbm = lgb.train(params, dtrain, num_boost_round,
                valid_sets=(dtrain, dvalid),
                valid_names=['train', 'valid'],
                verbose_eval=100,
                early_stopping_rounds=50)


# In[8]:


y_hat = gbm.predict(X_va)
score = rmse(y_va, y_hat)
print('rmse:', score)
feature_importance_ = zip(feature_names, gbm.feature_importance())
feature_importance_ = sorted(feature_importance_, key=lambda x: x[1], reverse=True)
print('\n'.join(('%s: %.2f' % x) for x in feature_importance_[:100]))


cv_pr = gbm.predict(X_te, num_iteration=gbm.best_iteration)
pr_hat = pd.Series(np.clip(cv_pr, 0, 1), name='deal_probability', index=te_index)

pr_hat_file = root+'results/lgb_base_pr_{}_{}.csv'.format(file_stamp, score)
pr_hat.to_csv(pr_hat_file, header=True)


X, X_te, feature_names = cut_feature(feature_importance_)
categorical = list(filter(lambda c: c.endswith('_cluster'),feature_names))


n_models = 2
x_score = []
final_cv_tr = np.zeros(len(y))
final_cv_pr = np.zeros(len(te_index))


params['learning_rate'] = 0.067
#params['feature_fraction'] = 0.86

for s in range(n_models):
    cv_tr = np.zeros(len(y))
    cv_pr = np.zeros(len(te_index))
    
    print(20 * '*' + 'lgb[{}]'.format(s) + 21 * '*')
    kf = kfold.split(X, round(y))
    best_trees = []
    fold_scores = []
    
    for i, (tr, va) in enumerate(kf):
        print(20 * '/' + 'lgb[{}]'.format(s), 'cv[{}]'.format(i) + 15 * '/')
        params['seed'] = s * i
        
        X_tr, X_va, y_tr, y_va = X[tr, :], X[va, :], y[tr], y[va]
        print(X_tr.shape, X_va.shape)
        dtrain = lgb.Dataset(X_tr, y_tr, feature_name=feature_names)
        dvalid = lgb.Dataset(X_va, y_va, feature_name=feature_names, reference=dtrain)
        gbm = lgb.train(params, dtrain, num_boost_round,
                        valid_sets=(dtrain, dvalid),
                        valid_names=['train', 'valid'],
                        verbose_eval=100,
                        early_stopping_rounds=50)
        best_trees.append(gbm.best_iteration)
        cv_pr += gbm.predict(X_te, num_iteration=gbm.best_iteration)
        cv_tr[va] += gbm.predict(X_va)
        score = rmse(y_va, cv_tr[va])
        print('lgb[{}]'.format(s), 'cv[{}]'.format(i), 'rmse:', score)
        fold_scores.append(score)
        feature_importance_ = zip(feature_names, gbm.feature_importance())
        feature_importance_ = sorted(feature_importance_, key=lambda x: x[1], reverse=True)
        pickle.dump(feature_importance_, open(root + 'results/lgb_20180519_{}_{}_feature_importance.pkl'.format(s, i), 'wb'))
        print('lgb[{}]'.format(s), 'cv[{}]'.format(i), 'importance features')
        print('\n'.join(('%s: %.2f' % x) for x in feature_importance_[:100]))
        print(50 * '/')
        
    cv_pr /= n_folds
    final_cv_tr += cv_tr
    final_cv_pr += cv_pr

    fold_score = rmse(y, cv_tr)
    x_score.append(fold_score)
    print('lgb[{}]'.format(s), 'fold score:', fold_score)
    print('lgb[{}]'.format(s), 'mean score:', rmse(y, cv_tr) / (s + 1.0), s + 1)
    print('lgb[{}]'.format(s), fold_scores)
    print('lgb[{}]'.format(s), best_trees, np.mean(best_trees))


# In[ ]:


print(50 * '-')
print('Fininsh CV')

print('\n'.join(['lgm[{}] score:{}'.format(*x) for x in enumerate(fold_scores)]))

final_avg_score = round(np.mean(x_score), 6)
final_std_score = round(np.std(x_score), 6)
print('avg score:%.6f' % final_avg_score)
print('std score:%.6f' % final_std_score)

pr_hat = pd.Series(np.clip(final_cv_pr / float(n_models), 0, 1), name='deal_probability', index=te_index)
pr_hat_file = root+'results/lgb_{}_pr_avg_{}_{}.csv'.format(n_models, file_stamp, final_avg_score)
pr_hat.to_csv(pr_hat_file, header=True)

tr_hat = pd.Series(np.clip(final_cv_tr / float(n_models), 0, 1), name='deal_probability', index=tr_index)
tr_hat_file = root+'results/lgb_{}_tr_avg_{}_{}.csv'.format(n_models, file_stamp, final_avg_score)
tr_hat.to_csv(tr_hat_file, header=True)

print('\n'.join([pr_hat_file, tr_hat_file]))


