
# coding: utf-8

# In[2]:

from __future__ import division
import numpy as np
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import time
import datetime
import warnings
warnings.filterwarnings('ignore')
import util


# In[3]:

def merge_feature(
    act_type_window,
    act_type_num_window,
    act_type_rate_window,
    act_type_row_stat_window,
    act_time_window,
    act_time_1type_window,
    act_ord_act_time_diff_window,
    action_sequence_time_diff_window,
    action_time_diff_234_56789_window,
    action_time_diff_stat_window
):
    util.log('Merge feature...')
    
    order_future_tr = pd.read_csv('../data/input/train/orderFuture_train.csv')
    order_future_te = pd.read_csv('../data/input/test/orderFuture_test.csv')

    user_profile = pd.read_csv('../data/output/feat/%s' % 'user_profile')
    train = pd.merge(order_future_tr, user_profile, on='userid', how='left')
    test = pd.merge(order_future_te, user_profile, on='userid', how='left')
    
    user_comment = pd.read_csv('../data/output/feat/%s' % 'user_comment')
    train = pd.merge(train, user_comment, on='userid', how='left')
    test = pd.merge(test, user_comment, on='userid', how='left')
    
    order_history = pd.read_csv('../data/output/feat/%s' % 'order_history')
    train = pd.merge(train, order_history, on='userid', how='left')
    test = pd.merge(test, order_history, on='userid', how='left')
    
#     order_history_last_w = pd.read_csv('../data/output/feat/%s' % 'order_history_last_w')
#     train = pd.merge(train, order_history_last_w, on='userid', how='left')
#     test = pd.merge(test, order_history_last_w, on='userid', how='left')
    
    action_type = pd.read_csv('../data/output/feat/%s' % 'action_type')
    train = pd.merge(train, action_type, on='userid', how='left')
    test = pd.merge(test, action_type, on='userid', how='left')
    
    action_type_based_on_time = pd.read_csv('../data/output/feat/%s' % 'action_type_based_on_time')
    train = pd.merge(train, action_type_based_on_time, on='userid', how='left')
    test = pd.merge(test, action_type_based_on_time, on='userid', how='left')
    
    util.log('act_type_window=' + str(act_type_window))
    window = act_type_window
    action_type_based_on_time_last_window = pd.read_csv('../data/output/feat/%s%d' % ('action_type_based_on_time_last_window', window))
    train = pd.merge(train, action_type_based_on_time_last_window, on='userid', how='left')
    test = pd.merge(test, action_type_based_on_time_last_window, on='userid', how='left')
    
    util.log('act_type_num_window=' + str(act_type_num_window))
    window = act_type_num_window
    action_type_num_based_on_time_last_window = pd.read_csv('../data/output/feat/%s%d' % ('action_type_num_based_on_time_last_window', window))
    train = pd.merge(train, action_type_num_based_on_time_last_window, on='userid', how='left')
    test = pd.merge(test, action_type_num_based_on_time_last_window, on='userid', how='left')
    
    util.log('act_type_rate_window=' + str(act_type_rate_window))
    window = act_type_rate_window
    action_type_rate_based_on_time_last_window = pd.read_csv('../data/output/feat/%s%d' % ('action_type_rate_based_on_time_last_window', window))
    train = pd.merge(train, action_type_rate_based_on_time_last_window, on='userid', how='left')
    test = pd.merge(test, action_type_rate_based_on_time_last_window, on='userid', how='left')
    
    util.log('act_type_row_stat_window=' + str(act_type_row_stat_window))
    window = act_type_row_stat_window
    action_type_row_stat_based_on_time_last_window_feat = pd.read_csv('../data/output/feat/%s%d' % ('action_type_row_stat_based_on_time_last_window', window))
    train = pd.merge(train, action_type_row_stat_based_on_time_last_window_feat, on='userid', how='left')
    test = pd.merge(test, action_type_row_stat_based_on_time_last_window_feat, on='userid', how='left')
    
#     util.log('action_num_window=' + str(action_num_window))
#     window = action_num_window
#     action_num_based_on_time_last_window = pd.read_csv('../data/output/feat/%s%d' % ('action_num_based_on_time_last_window', window))
#     train = pd.merge(train, action_num_based_on_time_last_window, on='userid', how='left')
#     test = pd.merge(test, action_num_based_on_time_last_window, on='userid', how='left')

#     util.log('action_type_num_window=' + str(action_type_num_window))
#     window = action_type_num_window
#     action_type_num_based_on_time_last_window = pd.read_csv('../data/output/feat/%s%d' % ('action_type_num_based_on_time_last_window', window))
#     train = pd.merge(train, action_type_num_based_on_time_last_window, on='userid', how='left')
#     test = pd.merge(test, action_type_num_based_on_time_last_window, on='userid', how='left')

    action_time_based_on_time = pd.read_csv('../data/output/feat/%s' % 'action_time_based_on_time')
    train = pd.merge(train, action_time_based_on_time, on='userid', how='left')
    test = pd.merge(test, action_time_based_on_time, on='userid', how='left')
    
    util.log('act_time_window=' + str(act_time_window))
    window = act_time_window
    action_time_based_on_time_last_window = pd.read_csv('../data/output/feat/%s%d' % ('action_time_based_on_time_last_window', window))
    train = pd.merge(train, action_time_based_on_time_last_window, on='userid', how='left')
    test = pd.merge(test, action_time_based_on_time_last_window, on='userid', how='left')
    
#     util.log('act_time_row_stat_window=' + str(act_time_row_stat_window))
#     window = act_time_row_stat_window
#     action_time_row_stat_based_on_time_last_window = pd.read_csv('../data/output/feat/%s%d' % ('action_time_row_stat_based_on_time_last_window', window))
#     train = pd.merge(train, action_time_row_stat_based_on_time_last_window, on='userid', how='left')
#     test = pd.merge(test, action_time_row_stat_based_on_time_last_window, on='userid', how='left')
 
#     util.log('action_time_diff2_window=' + str(action_time_diff2_window))
#     window = action_time_diff2_window
#     action_time_diff2_based_on_time_last_window = pd.read_csv('../data/output/feat/%s%d' % ('action_time_diff2_based_on_time_last_window', window))
#     train = pd.merge(train, action_time_diff2_based_on_time_last_window, on='userid', how='left')
#     test = pd.merge(test, action_time_diff2_based_on_time_last_window, on='userid', how='left')

    util.log('act_time_1type_window=%d' % act_time_1type_window)
    window = act_time_1type_window
    for ttype in [1, 5, 6, 7, 8, 9]:
        action_time_based_on_time_last_window_on_type = pd.read_csv('../data/output/feat/%s%d%s%d' % ('action_time_based_on_time_last_window', window, '_on_type', ttype))
        train = pd.merge(train, action_time_based_on_time_last_window_on_type, on='userid', how='left')
        test = pd.merge(test, action_time_based_on_time_last_window_on_type, on='userid', how='left')
    
#     util.log('action_time_2order_window=' + str(action_time_2order_window))
#     window = action_time_2order_window
#     action_time_2order_based_on_time_last_window = pd.read_csv('../data/output/feat/%s%d' % ('action_time_2order_based_on_time_last_window', window))
#     train = pd.merge(train, action_time_2order_based_on_time_last_window, on='userid', how='left')
#     test = pd.merge(test, action_time_2order_based_on_time_last_window, on='userid', how='left')

#     util.log('act_real_time_1type_window=%d' % act_real_time_1type_window)
#     window = act_real_time_1type_window
#     for ttype in [1, 5, 6, 7, 8, 9]:
#         action_real_time_based_on_time_last_window_on_type = pd.read_csv('../data/output/feat/%s%d%s%d' % ('action_real_time_based_on_time_last_window', window, '_on_type', ttype))
#         train = pd.merge(train, action_real_time_based_on_time_last_window_on_type, on='userid', how='left')
#         test = pd.merge(test, action_real_time_based_on_time_last_window_on_type, on='userid', how='left')

#     action_order_time_diff = pd.read_csv('../data/output/feat/%s' % 'action_order_time_diff')
#     train = pd.merge(train, action_order_time_diff, on='userid', how='left')
#     test = pd.merge(test, action_order_time_diff, on='userid', how='left')

#     order_last_order_ydm = pd.read_csv('../data/output/feat/%s' % 'order_last_order_ydm')
#     train = pd.merge(train, order_last_order_ydm, on='userid', how='left')
#     test = pd.merge(test, order_last_order_ydm, on='userid', how='left')

    order_type1_ydm = pd.read_csv('../data/output/feat/%s' % 'order_type1_ydm')
    train = pd.merge(train, order_type1_ydm, on='userid', how='left')
    test = pd.merge(test, order_type1_ydm, on='userid', how='left')

    util.log('act_ord_act_time_diff_window=' + str(act_ord_act_time_diff_window))
    window = act_ord_act_time_diff_window
    act_ord_act_time_diff_last_window = pd.read_csv('../data/output/feat/%s%d' % ('act_ord_act_time_diff_last_window', window))
    train = pd.merge(train, act_ord_act_time_diff_last_window, on='userid', how='left')
    test = pd.merge(test, act_ord_act_time_diff_last_window, on='userid', how='left')

#     util.log('act_ord_type1_act_time_diff_window=' + str(act_ord_type1_act_time_diff_window))
#     window = act_ord_type1_act_time_diff_window
#     act_ord_type1_act_time_diff_last_window = pd.read_csv('../data/output/feat/%s%d' % ('act_ord_type1_act_time_diff_last_window', window))
#     train = pd.merge(train, act_ord_type1_act_time_diff_last_window, on='userid', how='left')
#     test = pd.merge(test, act_ord_type1_act_time_diff_last_window, on='userid', how='left')

    util.log('action_sequence_time_diff_window=' + str(action_sequence_time_diff_window))
    window = action_sequence_time_diff_window
    action_sequence_time_diff_window = pd.read_csv('../data/output/feat/%s%d' % ('action_sequence_time_diff_window', window))
    train = pd.merge(train, action_sequence_time_diff_window, on='userid', how='left')
    test = pd.merge(test, action_sequence_time_diff_window, on='userid', how='left')

#     action_sequence_time_stat_last123 = pd.read_csv('../data/output/feat/%s' % 'action_sequence_time_stat_last123')
#     train = pd.merge(train, action_sequence_time_stat_last123, on='userid', how='left')
#     test = pd.merge(test, action_sequence_time_stat_last123, on='userid', how='left')

    util.log('action_time_diff_234_56789_window=' + str(action_time_diff_234_56789_window))
    window = action_time_diff_234_56789_window
    action_time_diff_234_56789_last_window = pd.read_csv('../data/output/feat/%s%d' % ('action_time_diff_234_56789_last_window', window))
    train = pd.merge(train, action_time_diff_234_56789_last_window, on='userid', how='left')
    test = pd.merge(test, action_time_diff_234_56789_last_window, on='userid', how='left')
    
#     action_stat_last_every_type = pd.read_csv('../data/output/feat/%s' % 'action_stat_last_every_type')
#     train = pd.merge(train, action_stat_last_every_type, on='userid', how='left')
#     test = pd.merge(test, action_stat_last_every_type, on='userid', how='left')

#     act_ord_before_type1_stat = pd.read_csv('../data/output/feat/%s' % 'act_ord_before_type1_stat')
#     train = pd.merge(train, act_ord_before_type1_stat, on='userid', how='left')
#     test = pd.merge(test, act_ord_before_type1_stat, on='userid', how='left')

    action_time_diff_stat = pd.read_csv('../data/output/feat/%s' % 'action_time_diff_stat')
    train = pd.merge(train, action_time_diff_stat, on='userid', how='left')
    test = pd.merge(test, action_time_diff_stat, on='userid', how='left')

    util.log('action_time_diff_stat_window=' + str(action_time_diff_stat_window))
    window = action_time_diff_stat_window
    action_time_diff_stat_last_window = pd.read_csv('../data/output/feat/%s%d' % ('action_time_diff_stat_last_window', window))
    train = pd.merge(train, action_time_diff_stat_last_window, on='userid', how='left')
    test = pd.merge(test, action_time_diff_stat_last_window, on='userid', how='left')
    
#     action_time_last_on_every_type = pd.read_csv('../data/output/feat/%s' % 'action_time_last_on_every_type')
#     train = pd.merge(train, action_time_last_on_every_type, on='userid', how='left')
#     test = pd.merge(test, action_time_last_on_every_type, on='userid', how='left')

    # bjw comment 中出现 order 中没有出现的为 1
    bjw_train = pd.read_csv('../data/output/feat/bjw/train_fea.csv')
    bjw_test = pd.read_csv('../data/output/feat/bjw/test_fea.csv')
    train = pd.merge(train, bjw_train, on='userid', how='left')
    test = pd.merge(test, bjw_test, on='userid', how='left')
    
    # 别人的开源特征，基于自己理解实现了一部分
    tryy = pd.read_csv('../data/output/feat/%s' % 'try')
    train = pd.merge(train, tryy, on='userid', how='left')
    test = pd.merge(test, tryy, on='userid', how='left')
    
    # bjw 的特征
    bjw_train = pd.read_csv('../data/output/feat/bjw/all_features_train.csv').drop(['Unnamed: 0', 'orderType'], axis=1)
    bjw_train.columns = ['userid' if i == 0 else i for i in range(len(bjw_train.columns))]
    bjw_test = pd.read_csv('../data/output/feat/bjw/all_features_test.csv').drop(['Unnamed: 0'], axis=1)
    bjw_test.columns = ['userid' if i == 0 else i for i in range(len(bjw_test.columns))]
    train = pd.merge(train, bjw_train, on='userid', how='left')
    test = pd.merge(test, bjw_test, on='userid', how='left')
    
#################################################################################################################
    
    # 用于交叉特征，使用之后会移除
    window = 1
    for ttype in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        action_real_time_based_on_time_last_window_on_type = pd.read_csv('../data/output/feat/%s%d%s%d' % ('action_real_time_based_on_time_last_window', window, '_on_type', ttype))
        train = pd.merge(train, action_real_time_based_on_time_last_window_on_type, on='userid', how='left')
        test = pd.merge(test, action_real_time_based_on_time_last_window_on_type, on='userid', how='left')

    train, test = cross_feature(train, test)
    
    train, test = drop_duplicate_column(train, test)
    
    train_feature = train.drop(['orderType'], axis = 1)
    train_label = train.orderType.values
    test_feature = test
    test_index = test.userid.values
    
    return train_feature, train_label, test_feature, test_index


# In[4]:

def cross_feature(train, test):
    util.log('Cross feature...')
    
    # 最近的 action 与最近的 order 的时间差
    train['act_last_time-ord_last_time'] = train['act_last_time'] - train['ord_last_time']
    train['act_last_time-ord_type0_time_max'] = train['act_last_time'] - train['ord_type0_time_max']
    train['act_last_time-ord_type1_time_max'] = train['act_last_time'] - train['ord_type1_time_max']
    test['act_last_time-ord_last_time'] = test['act_last_time'] - test['ord_last_time']
    test['act_last_time-ord_type0_time_max'] = test['act_last_time'] - test['ord_type0_time_max']
    test['act_last_time-ord_type1_time_max'] = test['act_last_time'] - test['ord_type1_time_max']
    
    # 最早的 action 与最早的 order 的时间差
    train['act_first_time-ord_first_time'] = train['act_first_time'] - train['ord_first_time']
    train['act_first_time-ord_type0_time_min'] = train['act_first_time'] - train['ord_type0_time_min']
    train['act_first_time-ord_type1_time_min'] = train['act_first_time'] - train['ord_type1_time_min']
    test['act_first_time-ord_first_time'] = test['act_first_time'] - test['ord_first_time']
    test['act_first_time-ord_type0_time_min'] = test['act_first_time'] - test['ord_type0_time_min']
    test['act_first_time-ord_type1_time_min'] = test['act_first_time'] - test['ord_type1_time_min']
    
    # 最近的 action 与最近的每一个 type 的 action 的时间差 + 最早的 action 与最早的每一个 type 的 action 的时间差
    for ttype in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        train['act_last_time-act_time(rank_1)(window_1)(type_%d)' % ttype] = train['act_last_time'] - train['act_time(rank_1)(window_1)(type_%d)' % ttype]
        train['act_first_time-act_time(rank_1)(window_1)(type_%d)' % ttype] = train['act_first_time'] - train['act_time(rank_1)(window_1)(type_%d)' % ttype]
        test['act_last_time-act_time(rank_1)(window_1)(type_%d)' % ttype] = test['act_last_time'] - test['act_time(rank_1)(window_1)(type_%d)' % ttype]
        test['act_first_time-act_time(rank_1)(window_1)(type_%d)' % ttype] = test['act_first_time'] - test['act_time(rank_1)(window_1)(type_%d)' % ttype]
        train = train.drop(['act_time(rank_1)(window_1)(type_%d)' % ttype], axis=1)
        test = test.drop(['act_time(rank_1)(window_1)(type_%d)' % ttype], axis=1)

    # 是否下过精品服务的单 * 最近的 action 的时间
    tmp = train['ord_num(type_1)'].copy()
    tmp[tmp > 1] = 1
    tmp = pd.get_dummies(tmp.fillna(-1))
    tmp.columns = ['has_ord_serv_nan', 'has_ord_serv_no', 'has_ord_serv_yes']
    train = pd.concat([train, tmp.mul(train['act_last_time'], axis=0)], axis=1)
    tmp = test['ord_num(type_1)'].copy()
    tmp[tmp > 1] = 1
    tmp = pd.get_dummies(tmp.fillna(-1))
    tmp.columns = ['has_ord_serv_nan', 'has_ord_serv_no', 'has_ord_serv_yes']
    test = pd.concat([test, tmp.mul(test['act_last_time'], axis=0)], axis=1)
    
    # 是否下过精品服务的单 * 每一个 type 的 action 的数量
    tmp = train['ord_num(type_1)'].copy()
    tmp[tmp > 1] = 1
    tmp = pd.get_dummies(tmp.fillna(-1))
    tmp.columns = ['has_ord_serv_nan', 'has_ord_serv_no', 'has_ord_serv_yes']
    for ttype in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        train = train.join(tmp.mul(train['act_num(type_%d)' % ttype], axis=0), rsuffix='*act_num(type_%d)' % ttype)
    tmp = test['ord_num(type_1)'].copy()
    tmp[tmp > 1] = 1
    tmp = pd.get_dummies(tmp.fillna(-1))
    tmp.columns = ['has_ord_serv_nan', 'has_ord_serv_no', 'has_ord_serv_yes']
    for ttype in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        test = test.join(tmp.mul(test['act_num(type_%d)' % ttype], axis=0), rsuffix='*act_num(type_%d)' % ttype)
        
#     # 最近的 order 与最近的每一个 type 的 action 的时间差 + 最早的 order 与最早的每一个 type 的 action 的时间差 （all/0/1）
#     for ttype in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
#         train['ord_last_time-act_time(rank_1)(window_1)(type_%d)' % ttype] = train['ord_last_time'] -  train['act_time(rank_1)(window_1)(type_%d)' % ttype]
#         train['ord_type0_time_max-act_time(rank_1)(window_1)(type_%d)' % ttype] = train['ord_type0_time_max'] - train['act_time(rank_1)(window_1)(type_%d)' % ttype]
#         train['ord_type1_time_max-act_time(rank_1)(window_1)(type_%d)' % ttype] = train['ord_type1_time_max'] - train['act_time(rank_1)(window_1)(type_%d)' % ttype]
#         train['ord_first_time-act_time(rank_1)(window_1)(type_%d)' % ttype] = train['ord_first_time'] - train['act_time(rank_1)(window_1)(type_%d)' % ttype]
#         train['ord_type0_time_min-act_time(rank_1)(window_1)(type_%d)' % ttype] = train['ord_type0_time_min'] - train['act_time(rank_1)(window_1)(type_%d)' % ttype]
#         train['ord_type1_time_min-act_time(rank_1)(window_1)(type_%d)' % ttype] = train['ord_type1_time_min'] - train['act_time(rank_1)(window_1)(type_%d)' % ttype]
#         test['ord_last_time-act_time(rank_1)(window_1)(type_%d)' % ttype] = test['ord_last_time'] -  test['act_time(rank_1)(window_1)(type_%d)' % ttype]
#         test['ord_type0_time_max-act_time(rank_1)(window_1)(type_%d)' % ttype] = test['ord_type0_time_max'] - test['act_time(rank_1)(window_1)(type_%d)' % ttype]
#         test['ord_type1_time_max-act_time(rank_1)(window_1)(type_%d)' % ttype] = test['ord_type1_time_max'] - test['act_time(rank_1)(window_1)(type_%d)' % ttype]
#         test['ord_first_time-act_time(rank_1)(window_1)(type_%d)' % ttype] = test['ord_first_time'] - test['act_time(rank_1)(window_1)(type_%d)' % ttype]
#         test['ord_type0_time_min-act_time(rank_1)(window_1)(type_%d)' % ttype] = test['ord_type0_time_min'] - test['act_time(rank_1)(window_1)(type_%d)' % ttype]
#         test['ord_type1_time_min-act_time(rank_1)(window_1)(type_%d)' % ttype] = test['ord_type1_time_min'] - test['act_time(rank_1)(window_1)(type_%d)' % ttype]
        
    return train, test


# In[5]:

def drop_duplicate_column(train, test):
    util.log('Drop duplicate column...')
    
    train = train.drop(['act_type(rank_1)(window6)'], axis=1)  # window9
    test = test.drop(['act_type(rank_1)(window6)'], axis=1)
    
    return train, test


# In[6]:

def lgb_cv(train_feature, train_label, params, folds, rounds):
    start = time.clock()
    print train_feature.columns
    dtrain = lgb.Dataset(train_feature, label=train_label)
    num_round = rounds
    print 'run cv: ' + 'round: ' + str(rounds)
    res = lgb.cv(params, dtrain, num_round, nfold=folds, verbose_eval=20, early_stopping_rounds=100)
    elapsed = (time.clock() - start)
    print 'Time used:', elapsed, 's'
    return len(res['auc-mean']), res['auc-mean'][len(res['auc-mean']) - 1]


def lgb_predict(train_feature, train_label, test_feature, rounds, params):
    dtrain = lgb.Dataset(train_feature, label=train_label)
    valid_sets = [dtrain]
    num_round = rounds
    model = lgb.train(params, dtrain, num_round, valid_sets, verbose_eval=50)
    predict = model.predict(test_feature)
    return model, predict


def store_result(test_index, pred, name):
    result = pd.DataFrame({'userid': test_index, 'orderType': pred})
    result.to_csv('../data/output/sub/' + name + '.csv', index=0, columns=['userid', 'orderType'])
    return result


# In[7]:

train_feature, train_label, test_feature, test_index = merge_feature(6, 6, 3, 6, 6, 6, 6, 6, 6, 3)
print train_feature.shape, train_label.shape, test_feature.shape


# In[8]:

config = {
    'rounds': 10000,
    'folds': 5
}

params_lgb = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'min_sum_hessian_in_leaf': 0.1,
    'learning_rate': 0.01,
    'verbosity': 2,
    'tree_learner': 'feature',
    'num_leaves': 128,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.9,
    'bagging_freq': 1,
    'num_threads': 16,
    'seed': 7
}


# In[9]:

params_lgb = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'min_child_weight': 1.5,
    'num_leaves': 2**5,
    'lambda_l2': 10,
    'subsample': 0.7,
    'colsample_bytree': 0.5,
    'colsample_bylevel': 0.5,
    'learning_rate': 0.01,
    'seed': 2017,
    'nthread': 12,
    'silent': True,
}


# In[10]:

iterations, best_score = lgb_cv(train_feature, train_label, params_lgb, config['folds'], config['rounds'])


# In[11]:

preds = 0
for s in range(7, 11):
    params_lgb['seed'] = s
    model, pred = lgb_predict(train_feature, train_label, test_feature, iterations, params_lgb)
    preds += pred
preds /= 4


# In[12]:

res = store_result(test_index, preds, '20180210-lgb-%f(r%d)' % (best_score, iterations))


# In[13]:

print("\n".join(("%s: %.2f" % x) for x in sorted(zip(train_feature.columns, model.feature_importance("gain")), key=lambda x: x[1], reverse=True)))


# In[ ]:




# In[ ]:

######################################### blending #########################################


# In[ ]:

test1 = pd.read_csv('../data/output/sub/bjw/result_addUserid_0125_1.csv')
test2 = pd.read_csv('../data/output/sub/20180203-lgb-0.966497(r1843).csv')
test3 = pd.read_csv('../data/output/sub/shawn_lgb_local9641_online9646.csv')
test4 = pd.read_csv('../data/output/sub/ym/lz96490.csv')
testa = pd.merge(test1, test2, on='userid', how='left')
testb = pd.merge(test3, test4, on='userid', how='left')
test = pd.merge(testa, testb, on='userid', how='left')


# In[ ]:

test['orderType'] = 0.5 * test['orderType_x_x'] + 0.3 * test['orderType_y_x'] + 0.1 * test['orderType_x_y'] + 0.1 * test['orderType_y_y']


# In[ ]:

test[['userid','orderType']].to_csv('../data/output/sub/blend/20180203-0.5bjw+0.3+0.1+0.1ym.csv',index=False)

