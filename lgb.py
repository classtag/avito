# coding: utf-8
"""
train lightgbm model
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

import gc
import pickle
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

root = '/kaggle/competitions/avito-demand-prediction/'

def load_matrix(name):
    return pickle.load(open(root + 'features/{}.pkl'.format(name), 'rb'))

daset_num_cols = load_matrix('daset_num_cols')
tr_c2v_matrix = load_matrix('tr_c2v_matrix')
tr_param_vec_matrix = load_matrix('tr_param_vec_matrix')
tr_txt_vec_matrix = load_matrix('tr_txt_vec_matrix')
tr_list = [daset_num_cols.loc[tr_index].values, tr_c2v_matrix, tr_param_vec_matrix, tr_txt_vec_matrix]
X = hstack(tr_list).tocsr()
dump_matrix(X, 'X')
del tr_list, tr_c2v_matrix,tr_param_vec_matrix,tr_txt_vec_matrix
gc.collect()

te_c2v_matrix = load_matrix('te_c2v_matrix')
te_param_vec_matrix = load_matrix('te_param_vec_matrix')
te_txt_vec_matrix = load_matrix('te_txt_vec_matrix')
te_list = [daset_num_cols.loc[te_index].values, te_c2v_matrix, te_param_vec_matrix, te_txt_vec_matrix]
X_te = hstack(te_list).tocsr()
dump_matrix(X_te, 'X_te')
del te_list, te_c2v_matrix, te_param_vec_matrix, te_txt_vec_matrix
gc.collect()

print('Load dataset.', feature_dump)

print('Load dataset')
tr_index = load_matrix('tr_index')
te_index = load_matrix('te_index')
feature_names = load_matrix('feature_names')

print('dataset shape:', X.shape, X_te.shape, len(y))
print('feature names:\n' + '\n'.join(feature_names))

print('Modeling')
num_boost_round = 20000
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 15,
    'max_bin': 256,
    'feature_fraction': 0.6,
    'min_child_samples': 10,
    'min_child_weight': 150,
    'min_split_gain': 0,
    'subsample': 0.9,
    'drop_rate': 0.1,
    'max_drop': 50,
    'verbosity': 0
}
n_folds = 5
kfold = StratifiedKFold(n_splits=n_folds, shuffle=False, random_state=218)
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

n_models = 1
x_score = []
final_cv_tr = np.zeros(len(y))
final_cv_pr = np.zeros(len(te_index))
for s in range(n_models):
    cv_tr = np.zeros(len(y))
    cv_pr = np.zeros(len(te_index))
    
    print(20 * '*' + 'lgb[{}]'.format(s) + 21 * '*')
    kf = kfold.split(X, round(y))
    best_trees = []
    fold_scores = []
    for i, (tr, va) in enumerate(kf):
        print(20 * '/' + 'lgb[{}]'.format(s), 'cv[{}]'.format(i) + 15 * '/')
        X_tr, X_va, y_tr, y_va = X[tr, :], X[va, :], y[tr], y[va]
        print(X_tr.shape, X_va.shape)
        dtrain = lgb.Dataset(X_tr, y_tr)
        dvalid = lgb.Dataset(X_va, y_va, reference=dtrain)
        params['seed'] = s * i
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
        feature_importance_ = zip(feature_names, gbm.feature_importance("gain"))
        feature_importance_ = sorted(feature_importance_, key=lambda x: x[1], reverse=True)
        pickle.dump(feature_importance_, open(root + 'results/lgb_{}_{}_feature_importance.pkl'.format(s, i), 'wb'))
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


print(50 * '-')
print('Fininsh CV')

file_stamp = datetime.now().strftime('%m%d%H%M')
print('\n'.join(['lgm[{}] score:{}'.format(*x) for x in enumerate(fold_scores)]))

final_avg_score = round(np.mean(x_score), 6)
final_std_score = round(np.std(x_score), 6)
print('avg score:%.6f' % final_avg_score)
print('std score:%.6f' % final_std_score)

pr_hat = pd.Series(np.clip(np.expm1(final_cv_pr / float(n_models)), 0, 1), name='deal_probability', index=te_index)
pr_hat_file = root+'results/lgb_{}_pr_avg_{}_{}.csv'.format(n_models, file_stamp, final_avg_score)
pr_hat.to_csv(pr_hat_file, header=True)

tr_hat = pd.Series(np.clip(np.expm1(final_cv_tr / float(n_models)), 0, 1), name='deal_probability', index=tr_index)
tr_hat_file = root+'results/lgb_{}_tr_avg_{}_{}.csv'.format(n_models, file_stamp, final_avg_score)
tr_hat.to_csv(tr_hat_file, header=True)

print('\n'.join([pr_hat_file, tr_hat_file]))
