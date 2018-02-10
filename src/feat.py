
# coding: utf-8

# In[ ]:

from __future__ import division
import numpy as np
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import time
import datetime
import sys
import math
import warnings
warnings.filterwarnings('ignore')
import util


# In[ ]:

def get_user_profile_feature(df):
    df = df.copy()

    mydf = df[['userid']]
    le = preprocessing.LabelEncoder()
    mydf['gender'] = le.fit_transform(df['gender'])

    mydf['province'] = le.fit_transform(df['province'])

    mydf['age'] = le.fit_transform(df['age'])

    return mydf


# In[ ]:

def get_user_comment_feature(df):
    df = df.copy()
    
    mydf = df[['userid']].drop_duplicates().reset_index(drop=True)

    com_rating = df.groupby('userid')['rating'].agg(['sum', 'count']).reset_index()
    com_rating.columns = [i if i == 'userid' else 'com_rating_' + i for i in com_rating.columns]

    mydf = pd.merge(mydf, com_rating, on='userid', how='left')
    
    return mydf


# In[ ]:

def get_order_history_feature(df):
    df = df.copy()

    mydf = df[['userid']].drop_duplicates().reset_index(drop=True)

    # type 为 0 和 1 的订单的数量和比率 + 总订单数
    ord_hist_ord = df.groupby('userid')['orderType'].agg(['sum', 'count']).reset_index()
    ord_hist_ord.columns = ['userid', 'ord_num(type_1)', 'ord_num']
    ord_hist_ord['ord_num(type_0)'] = ord_hist_ord['ord_num'] - ord_hist_ord['ord_num(type_1)']
    ord_hist_ord['ord_rate(type_1)'] = ord_hist_ord['ord_num(type_1)'] / ord_hist_ord['ord_num']
    ord_hist_ord['ord_rate(type_0)'] = ord_hist_ord['ord_num(type_0)'] / ord_hist_ord['ord_num']

    # city, country, continent 的数量
    addr_count = df.groupby('userid')['city', 'country', 'continent'].count().reset_index()
    addr_count.columns = ['userid', 'city_num', 'country_num', 'continent_num']

    # type 为 1 的 city, country, continent 的数量
    addr_count_pos = df[df['orderType'] == 1].groupby('userid')['city', 'country', 'continent'].count().reset_index()
    addr_count_pos.columns = ['userid', 'city_num(type_1)', 'country_num(type_1)', 'continent_num(type_1)']

    # 每个 country 的订单数量
    lb = preprocessing.LabelBinarizer()
    tmp = lb.fit_transform(df['country'])
    tmp_col = ['country_' + str(i) for i in range(tmp.shape[1])]
    tmp = pd.DataFrame(tmp, columns=tmp_col)
    tmp['userid'] = df['userid'].values
    country = tmp.groupby('userid')[tmp_col].agg(['sum']).reset_index()

    # 每个 continent 的订单数量
    lb = preprocessing.LabelBinarizer()
    tmp = lb.fit_transform(df['continent'])
    tmp_col = ['continent_' + str(i) for i in range(tmp.shape[1])]
    tmp = pd.DataFrame(tmp, columns=tmp_col)
    tmp['userid'] = df['userid'].values
    continent = tmp.groupby('userid')[tmp_col].agg(['sum']).reset_index()
    
    # 最后一次的 order
    last_ord = df.groupby('userid').apply(lambda x: x.sort_values('orderTime', ascending=False).head(1)).reset_index(drop=True)[['userid', 'orderid', 'orderTime', 'orderType']]
    last_ord.columns = ['userid', 'ord_last_id', 'ord_last_time', 'ord_last_type']
    
    # 第一次的 order
    first_ord = df.groupby('userid').apply(lambda x: x.sort_values('orderTime', ascending=True).head(1)).reset_index(drop=True)[['userid', 'orderid', 'orderTime', 'orderType']]
    first_ord.columns = ['userid', 'ord_first_id', 'ord_first_time', 'ord_first_type']
    
    # type 分别为 0/1 的订单的时间的统计
    for t in [0, 1]:
        ord_time_stat = df[df['orderType'] == t].groupby('userid')['orderTime'].agg([min, max, np.ptp, np.mean, np.median, np.std]).reset_index()
        ord_time_stat.columns = [i if i == 'userid' else 'ord_type%d_time_%s' % (t, i) for i in ord_time_stat.columns]
        mydf = pd.merge(mydf, ord_time_stat, on='userid', how='left')
    
    mydf = pd.merge(mydf, ord_hist_ord, on='userid', how='left')
    mydf = pd.merge(mydf, addr_count, on='userid', how='left')
    mydf = pd.merge(mydf, addr_count_pos, on='userid', how='left')
    mydf = pd.merge(mydf, country, on='userid', how='left')
    mydf = pd.merge(mydf, continent, on='userid', how='left')
    mydf = pd.merge(mydf, last_ord, on='userid', how='left')
    mydf = pd.merge(mydf, first_ord, on='userid', how='left')
        
    return mydf


# In[ ]:

def get_order_history_last_w_feature(df):
    df = df.copy()

    mydf = df[['userid']].drop_duplicates().reset_index(drop=True)

    # 最后 w 次订单的统计
    for w in [2, 3, 4]:
        util.log(w)
        
        last_order = df.groupby('userid').apply(lambda x: x.sort_values('orderTime', ascending=False).head(w)).reset_index(drop=True)[['userid', 'orderTime', 'orderType']]
        last_order.columns = ['userid', 'ord_last_time', 'ord_last_type']
        
        ord_last_time_stat = last_order.groupby('userid')['ord_last_time'].agg([min, max, np.ptp, np.mean, np.median, np.std]).reset_index()
        ord_last_time_stat.columns = [i if i == 'userid' else 'ord_last%d_time_%s' % (w, i) for i in ord_last_time_stat.columns]
        
        ord_last_type_stat = last_order.groupby('userid')['ord_last_type'].agg(['count', sum]).reset_index()
        ord_last_type_stat.columns = [i if i == 'userid' else 'ord_last%d_type_%s' % (w, i) for i in ord_last_type_stat.columns]
        
        mydf = pd.merge(mydf, ord_last_time_stat, on='userid', how='left')
        mydf = pd.merge(mydf, ord_last_type_stat, on='userid', how='left')

    return mydf


# In[ ]:

def get_action_type_feature(df):
    df = df.copy()

    mydf = df[['userid']].drop_duplicates().reset_index(drop=True)

    # 每个用户的 action 和 actionType 的数量
    act_num = df.groupby(['userid', 'actionType']).size().reset_index().groupby('userid')[0].agg([sum, len]).reset_index()
    act_num.columns = ['userid', 'act_num', 'act_type_num']

    # 每个类别的数量
    act_type_num = df.groupby(['userid', 'actionType']).size().unstack().reset_index()
    act_type_num.columns = [i if i == 'userid' else 'act_num(type_' + str(i) + ')' for i in act_type_num.columns]

    mydf = pd.merge(mydf, act_num, on='userid', how='left')
    mydf = pd.merge(mydf, act_type_num, on='userid', how='left')

    return mydf


# In[ ]:

def get_action_type_based_on_time_feature(df):
    df = df.copy()

    mydf = df[['userid']].drop_duplicates().reset_index(drop=True)

    # 最近的一次 action 的 type
    act_last_type = df.groupby('userid').apply(lambda x: x.sort_values('actionTime', ascending=False).head(1)).reset_index(drop=True)[['userid', 'actionType']]
    act_last_type.columns = ['userid', 'act_last_type']
    
    # 最早的一次 action 的 type
    act_first_type = df.groupby('userid').apply(lambda x: x.sort_values('actionTime', ascending=True).head(1)).reset_index(drop=True)[['userid', 'actionType']]
    act_first_type.columns = ['userid', 'act_first_type']

    mydf = pd.merge(mydf, act_last_type, on='userid', how='left')
    mydf = pd.merge(mydf, act_first_type, on='userid', how='left')

    return mydf


# In[ ]:

def get_action_type_based_on_time_last_window_feature(df, window):
    df = df.copy()

    mydf = df[['userid']].drop_duplicates().reset_index(drop=True)

    tmp = df.groupby('userid').apply(lambda x: x.sort_values('actionTime', ascending=False).head(window)).reset_index(drop=True)
    tmp['act_time_rank'] = tmp.groupby('userid')['actionTime'].rank(method = 'first', ascending=False).astype(int)

    # type 的差值
    act_type = tmp.pivot('userid', 'act_time_rank', 'actionType')
    act_type = act_type[act_type.columns[::-1]]
    act_type_diff = act_type.diff(1, axis=1)
    act_type_diff = act_type_diff.iloc[:, 1:].reset_index()
    act_type_diff.columns = [i if i == 'userid' else 'act_type_diff(' + str(i) + '-' + str(i + 1) + ')(window_' + str(window) + ')' for i in act_type_diff.columns]

    mydf = pd.merge(mydf, act_type_diff, on='userid', how='left')

    return mydf


# In[ ]:

def get_action_type_num_based_on_time_last_window_feature(df, window):
    df = df.copy()

    mydf = df[['userid']].drop_duplicates().reset_index(drop=True)

    tmp = df.groupby('userid').apply(lambda x: x.sort_values('actionTime', ascending=False).head(window)).reset_index(drop=True)
    tmp['act_time_rank'] = tmp.groupby('userid')['actionTime'].rank(method = 'first', ascending=False).astype(int)

    # 每个类别的数量
    act_num_in_window = tmp.groupby(['userid', 'actionType']).size().unstack().reset_index()
    act_num_in_window.columns = [i if i == 'userid' else 'act_num(type_' + str(i) + ')(window_' + str(window) + ')' for i in act_num_in_window.columns]
    
    mydf = pd.merge(mydf, act_num_in_window, on='userid', how='left')

    return mydf


# In[ ]:

def get_action_type_rate_based_on_time_last_window_feature(df, window):
    df = df.copy()

    mydf = df[['userid']].drop_duplicates().reset_index(drop=True)

    tmp = df.groupby('userid').apply(lambda x: x.sort_values('actionTime', ascending=False).head(window)).reset_index(drop=True)
    tmp['act_time_rank'] = tmp.groupby('userid')['actionTime'].rank(method = 'first', ascending=False).astype(int)

    # 每个类别的列级别的比率
    act_column_rate_in_window = tmp.groupby(['userid', 'actionType']).size().unstack().apply(lambda x: x / np.sum(x)).reset_index()
    act_column_rate_in_window.columns = [i if i == 'userid' else 'act_column_rate(type_' + str(i) + ')(window_' + str(window) + ')' for i in act_column_rate_in_window.columns]

    # 每个类别的行级别的比率
    act_row_rate_in_window = tmp.groupby(['userid', 'actionType']).size().unstack().apply((lambda x: x / np.sum(x)), axis=1).reset_index()
    act_row_rate_in_window.columns = [i if i == 'userid' else 'act_row_rate(type_' + str(i) + ')(window_' + str(window) + ')' for i in act_row_rate_in_window.columns]
    
    mydf = pd.merge(mydf, act_column_rate_in_window, on='userid', how='left')
    mydf = pd.merge(mydf, act_row_rate_in_window, on='userid', how='left')

    return mydf


# In[ ]:

def get_action_type_row_stat_based_on_time_last_window_feature(df, window):
    df = df.copy()

    mydf = df[['userid']].drop_duplicates().reset_index(drop=True)

    tmp = df.groupby('userid').apply(lambda x: x.sort_values('actionTime', ascending=False).head(window)).reset_index(drop=True)
    tmp['act_time_rank'] = tmp.groupby('userid')['actionTime'].rank(method = 'first', ascending=False).astype(int)

    # 最近的 type 值 + 行级别的统计值
    act_type = tmp.pivot('userid', 'act_time_rank', 'actionType')
    act_type.columns = ['act_type(rank_' + str(i) + ')(window' + str(window) + ')' for i in act_type.columns]
    for i in ['min', 'max', 'mean', 'median', 'std', 'sum', np.ptp]:
        act_type['act_row_type_' + i + '(window_' + str(window) + ')' if type(i) == str else 'act_row_type_' + i.func_name + '(window_' + str(window) + ')'] = act_type.apply(i, axis=1)
    act_type = act_type.reset_index()
    
    mydf = pd.merge(mydf, act_type, on='userid', how='left')

    return mydf


# In[ ]:

def get_action_num_based_on_time_last_window_feature(df, window):
    df = df.copy()

    mydf = df[['userid']].drop_duplicates().reset_index(drop=True)

    tmp = df.groupby('userid').apply(lambda x: x.sort_values('actionTime', ascending=False).head(window)).reset_index(drop=True)

    # action 的数量
    act_num = tmp.groupby('userid').size().reset_index()
    act_num.columns = ['userid', 'act_num(window_%d)' % window]
    
    mydf = pd.merge(mydf, act_num, on='userid', how='left')

    return mydf


# In[ ]:

def get_action_type_num_based_on_time_last_window_feature(df, window):
    df = df.copy()

    mydf = df[['userid']].drop_duplicates().reset_index(drop=True)

    tmp = df.groupby('userid').apply(lambda x: x.sort_values('actionTime', ascending=False).head(window)).reset_index(drop=True)

    # type 的数量
    act_type_num = tmp.groupby(['userid', 'actionType']).size().reset_index().groupby('userid')[0].agg([len]).reset_index()
    act_type_num.columns = ['userid', 'act_type_num(window_%d)' % window]
    
    mydf = pd.merge(mydf, act_type_num, on='userid', how='left')

    return mydf


# In[ ]:

def get_action_time_based_on_time_feature(df):
    df = df.copy()

    mydf = df[['userid']].drop_duplicates().reset_index(drop=True)

    # 最近的一次 action 的 time
    act_last_time = df.groupby('userid').apply(lambda x: x.sort_values('actionTime', ascending=False).head(1)).reset_index(drop=True)[['userid', 'actionTime']]
    act_last_time.columns = ['userid', 'act_last_time']
    
    # 最早的一次 action 的 time
    act_first_time = df.groupby('userid').apply(lambda x: x.sort_values('actionTime', ascending=True).head(1)).reset_index(drop=True)[['userid', 'actionTime']]
    act_first_time.columns = ['userid', 'act_first_time']
    
    mydf = pd.merge(mydf, act_last_time, on='userid', how='left')
    mydf = pd.merge(mydf, act_first_time, on='userid', how='left')
    
    mydf['act_time_last-first'] = mydf['act_last_time'] - mydf['act_first_time']

    return mydf


# In[ ]:

def get_action_time_based_on_time_last_window_feature(df, window):
    df = df.copy()

    mydf = df[['userid']].drop_duplicates().reset_index(drop=True)
    
    tmp = df.groupby('userid').apply(lambda x: x.sort_values('actionTime', ascending=False).head(window)).reset_index(drop=True)
    tmp['act_time_rank'] = tmp.groupby('userid')['actionTime'].rank(method = 'first', ascending=False).astype(int)
    
    # time 的差值
    act_time = tmp.pivot('userid', 'act_time_rank', 'actionTime')
    act_time = act_time[act_time.columns[::-1]]
    act_time_diff = act_time.diff(1, axis=1)
    act_time_diff = act_time_diff.iloc[:, 1:].reset_index()
    act_time_diff.columns = [i if i == 'userid' else 'act_time_diff(' + str(i) + '-' + str(i + 1) + ')(window_' + str(window) + ')' for i in act_time_diff.columns]

    mydf = pd.merge(mydf, act_time_diff, on='userid', how='left')
    
    return mydf


# In[ ]:

def get_action_time_row_stat_based_on_time_last_window_feature(df, window):
    df = df.copy()

    mydf = df[['userid']].drop_duplicates().reset_index(drop=True)
    
    tmp = df.groupby('userid').apply(lambda x: x.sort_values('actionTime', ascending=False).head(window)).reset_index(drop=True)
    tmp['act_time_rank'] = tmp.groupby('userid')['actionTime'].rank(method = 'first', ascending=False).astype(int)
    
    # 最近的 time 值 + 行级别的统计值
    act_time = tmp.pivot('userid', 'act_time_rank', 'actionTime')
    act_time.columns = ['act_time(rank_' + str(i) + ')(window' + str(window) + ')' for i in act_time.columns]
    for i in ['min', 'max', 'mean', 'median', 'std', 'sum', np.ptp]:
        act_time['act_row_time_' + i + '(window_' + str(window) + ')' if type(i) == str else 'act_row_time_' + i.func_name + '(window_' + str(window) + ')'] = act_time.apply(i, axis=1)
    act_time = act_time.reset_index()

    mydf = pd.merge(mydf, act_time, on='userid', how='left')
    
    return mydf


# In[ ]:

def get_action_time_diff2_based_on_time_last_window_feature(df, window):
    df = df.copy()

    mydf = df[['userid']].drop_duplicates().reset_index(drop=True)
    
    tmp = df.groupby('userid').apply(lambda x: x.sort_values('actionTime', ascending=False).head(window)).reset_index(drop=True)
    tmp['act_time_rank'] = tmp.groupby('userid')['actionTime'].rank(method = 'first', ascending=False).astype(int)
    
    # time 的差值
    act_time = tmp.pivot('userid', 'act_time_rank', 'actionTime')
    act_time = act_time[act_time.columns[::-1]]
    act_time_diff2 = act_time.diff(2, axis=1)    # need test
    act_time_diff2 = act_time_diff2.iloc[:, 2:].reset_index()
    act_time_diff2.columns = [i if i == 'userid' else 'act_time_diff2(' + str(i) + '-' + str(i + 2) + ')(window_' + str(window) + ')' for i in act_time_diff2.columns]

    mydf = pd.merge(mydf, act_time_diff2, on='userid', how='left')
    
    return mydf


# In[ ]:

def get_action_time_based_on_time_last_window_on_type_feature(df, window, ttype):
    df = df.copy()

    mydf = df[['userid']].drop_duplicates().reset_index(drop=True)
    
    tmp = df[df['actionType'] == ttype].groupby('userid').apply(lambda x: x.sort_values('actionTime', ascending=False).head(window)).reset_index(drop=True)
    tmp['act_time_rank'] = tmp.groupby('userid')['actionTime'].rank(method = 'first', ascending=False).astype(int)
    
    # 特定 type 的 action 的 time 的差值
    act_time = tmp.pivot('userid', 'act_time_rank', 'actionTime')
    act_time = act_time[act_time.columns[::-1]]
    act_time_diff = act_time.diff(1, axis=1)
    act_time_diff = act_time_diff.iloc[:, 1:].reset_index()
    act_time_diff.columns = [i if i == 'userid' else 'act_time_diff(%d-%d)(window_%d)(type_%d)' % (i, i+1, window, ttype) for i in act_time_diff.columns]

    mydf = pd.merge(mydf, act_time_diff, on='userid', how='left')
    
    return mydf


# In[ ]:

def get_action_time_2order_based_on_time_last_window_feature(df, window):
    df = df.copy()

    mydf = df[['userid']].drop_duplicates().reset_index(drop=True)
    
    tmp = df.groupby('userid').apply(lambda x: x.sort_values('actionTime', ascending=False).head(window)).reset_index(drop=True)
    tmp['act_time_rank'] = tmp.groupby('userid')['actionTime'].rank(method = 'first', ascending=False).astype(int)
    
    # 特定 type 的 action 的 time 的差值
    act_time = tmp.pivot('userid', 'act_time_rank', 'actionTime')
    act_time = act_time[act_time.columns[::-1]]
    act_time_diff_2order = act_time.diff(1, axis=1).diff(1, axis=1)
    act_time_diff_2order = act_time_diff_2order.iloc[:, 2:].reset_index()
    act_time_diff_2order.columns = [i if i == 'userid' else 'act_time_diff_2order(%d-%d)(window_%d)' % (i, i+1, window) for i in act_time_diff_2order.columns]

    mydf = pd.merge(mydf, act_time_diff_2order, on='userid', how='left')
    
    return mydf


# In[ ]:

def get_action_real_time_based_on_time_last_window_on_type_feature(df, window, ttype):
    df = df.copy()

    mydf = df[['userid']].drop_duplicates().reset_index(drop=True)
    
    tmp = df[df['actionType'] == ttype].groupby('userid').apply(lambda x: x.sort_values('actionTime', ascending=False).head(window)).reset_index(drop=True)
    tmp['act_time_rank'] = tmp.groupby('userid')['actionTime'].rank(method = 'first', ascending=False).astype(int)
    
    # 特定的 type 的 action 的最近的 time 值
    act_time = tmp.pivot('userid', 'act_time_rank', 'actionTime').reset_index()
    act_time.columns = [i if i == 'userid' else 'act_time(rank_%d)(window_%d)(type_%d)' % (i, window, ttype) for i in act_time.columns]

    mydf = pd.merge(mydf, act_time, on='userid', how='left')
    
    return mydf


# In[ ]:

def get_act_ord_time_diff_feature(act, oord):
    act = act.copy()
    oord = oord.copy()

    mydf = oord[['userid']].drop_duplicates().reset_index(drop=True)

    ord_time = oord.groupby('userid')['orderTime'].max().reset_index()
    act = pd.merge(act, ord_time, on='userid', how='left')  # fillna?
    act['act_time-ord_time'] = act['actionTime'] - act['orderTime']
    act_ord_time_diff = act[act['act_time-ord_time'] > 0].groupby('userid').size().reset_index()
    act_ord_time_diff.columns = ['userid', 'act_ord_time_diff_gt0_count']

    mydf = pd.merge(mydf, act_ord_time_diff, on='userid', how='left')
    
    return mydf


# In[ ]:

def get_order_last_order_ydm_feature(df):
    df = df.copy()

    mydf = df[['userid']].drop_duplicates().reset_index(drop=True)

    tmp = df.groupby('userid').apply(lambda x: x.sort_values('orderTime', ascending=False).head(1)).reset_index(drop=True)

    mydf['ord_last_ord_year'] = pd.to_datetime(tmp['orderTime'], unit='s').dt.year
    mydf['ord_last_ord_month'] = pd.to_datetime(tmp['orderTime'], unit='s').dt.month
    mydf['ord_last_ord_day'] = pd.to_datetime(tmp['orderTime'], unit='s').dt.day

    return mydf


# In[ ]:

def get_order_type1_ydm_feature(df):
    df = df.copy()

    mydf = df[['userid']].drop_duplicates().reset_index(drop=True)

    # 最近一次的 type 为 1 的订单的年月日
    tmp = df[df['orderType'] == 1].groupby('userid').apply(lambda x: x.sort_values('orderTime', ascending=False).head(1)).reset_index(drop=True)
    mydf['ord_last_type1_year'] = pd.to_datetime(tmp['orderTime'], unit='s').dt.year
    mydf['ord_last_type1_month'] = pd.to_datetime(tmp['orderTime'], unit='s').dt.month
    mydf['ord_last_type1_day'] = pd.to_datetime(tmp['orderTime'], unit='s').dt.day
    
    # 最早一次的 type 为 1 的订单的年月日
    tmp = df[df['orderType'] == 1].groupby('userid').apply(lambda x: x.sort_values('orderTime', ascending=True).head(1)).reset_index(drop=True)
    mydf['ord_first_type1_year'] = pd.to_datetime(tmp['orderTime'], unit='s').dt.year
    mydf['ord_first_type1_month'] = pd.to_datetime(tmp['orderTime'], unit='s').dt.month
    mydf['ord_first_type1_day'] = pd.to_datetime(tmp['orderTime'], unit='s').dt.day

    return mydf


# In[ ]:

def get_act_ord_act_time_diff_last_window_feature(act, oord, window):
    act = act.copy()
    oord = oord.copy()

    mydf = oord[['userid']].drop_duplicates().reset_index(drop=True)

    ord_time = oord.groupby('userid')['orderTime'].max().reset_index()
    act = pd.merge(act, ord_time, on='userid', how='left')

    df = act[act['actionTime'] < act['orderTime']]

    tmp = df.groupby('userid').apply(lambda x: x.sort_values('actionTime', ascending=False).head(window)).reset_index(drop=True)
    tmp['act_time_rank'] = tmp.groupby('userid')['actionTime'].rank(method = 'first', ascending=False).astype(int)

    # 最后一次订单之前的 action 的 time 的差值
    act_time = tmp.pivot('userid', 'act_time_rank', 'actionTime')
    act_time = act_time[act_time.columns[::-1]]
    act_time_diff = act_time.diff(1, axis=1)
    act_time_diff = act_time_diff.iloc[:, 1:].reset_index()
    act_time_diff.columns = [i if i == 'userid' else 'act_ord_act_time_diff(' + str(i) + '-' + str(i + 1) + ')(window_' + str(window) + ')' for i in act_time_diff.columns]

    mydf = pd.merge(mydf, act_time_diff, on='userid', how='left')
    
    return mydf


# In[ ]:

def get_act_ord_type1_act_time_diff_last_window_feature(act, oord, window):
    act = act.copy()
    oord = oord.copy()

    mydf = oord[['userid']].drop_duplicates().reset_index(drop=True)

    ord_time = oord[oord['orderType'] == 1].groupby('userid')['orderTime'].max().reset_index()
    act = pd.merge(act, ord_time, on='userid', how='left')

    df = act[act['actionTime'] < act['orderTime']]

    tmp = df.groupby('userid').apply(lambda x: x.sort_values('actionTime', ascending=False).head(window)).reset_index(drop=True)
    tmp['act_time_rank'] = tmp.groupby('userid')['actionTime'].rank(method = 'first', ascending=False).astype(int)

    # 最后一次精品订单之前的 action 的 time 的差值
    act_time = tmp.pivot('userid', 'act_time_rank', 'actionTime')
    act_time = act_time[act_time.columns[::-1]]
    act_time_diff = act_time.diff(1, axis=1)
    act_time_diff = act_time_diff.iloc[:, 1:].reset_index()
    act_time_diff.columns = [i if i == 'userid' else 'act_ord_type1_act_time_diff(' + str(i) + '-' + str(i + 1) + ')(window_' + str(window) + ')' for i in act_time_diff.columns]

    mydf = pd.merge(mydf, act_time_diff, on='userid', how='left')
    
    return mydf


# In[ ]:

def get_action_sequence_time_diff_feature(df):
    df = df.sort_values(by=['userid', 'actionTime'], ascending=[True, False]).copy().reset_index(drop=True)

    mydf = df[['userid']].drop_duplicates().reset_index(drop=True)

    df['actionTimee'] = pd.to_datetime(df['actionTime'], unit='s')
    df['actionTimeDiff'] = df['actionTime'].diff()

    counter = 1
    last_userid = df.iloc[0, 0]
    seq_list = []
    for i, r in df[['userid', 'actionTimeDiff']].iterrows():
        if i % 500000 == 0:
            util.log(i)
        if r.userid != last_userid:
            counter = 1
            seq_list.append(counter)
            last_userid = r.userid
        elif (r.actionTimeDiff <= 0 and r.actionTimeDiff >= -600 or math.isnan(r.actionTimeDiff)) and r.userid == last_userid:
            seq_list.append(counter)
        else:
            counter += 1
            seq_list.append(counter)
    df['actionSeq'] = pd.Series(seq_list)
    
    # 基于10分钟分块（时差低于10分钟的行为为一部分），每个块的时差
    seq_time_max = df.groupby(['userid', 'actionSeq'])['actionTime'].max().unstack()
    seq_time_diff = seq_time_max.diff(1, axis=1)
    for window in [2,3,4,5,6,7,10,15]:
        tmp = seq_time_diff.iloc[:, 1:(window+1)]
        tmp.columns = ['act_seq_time_diff(%d-%d)(window_%d)' % (i, i-1, window) for i in tmp.columns]
        tmp = tmp.reset_index()
        data = pd.merge(mydf, tmp, on='userid', how='left')
        util.log('window=%d' % window)
        data.to_csv('../data/output/feat/%s%d' % ('action_sequence_time_diff_window', window), index=False)


# In[ ]:

def get_action_sequence_time_stat_feature(df):
    df = df.sort_values(by=['userid', 'actionTime'], ascending=[True, False]).copy().reset_index(drop=True)

    mydf = df[['userid']].drop_duplicates().reset_index(drop=True)

    df['actionTimee'] = pd.to_datetime(df['actionTime'], unit='s')
    df['actionTimeDiff'] = df['actionTime'].diff()

    counter = 1
    last_userid = df.iloc[0, 0]
    seq_list = []
    for i, r in df[['userid', 'actionTimeDiff']].iterrows():
        if i % 500000 == 0:
            util.log(i)
        if r.userid != last_userid:
            counter = 1
            seq_list.append(counter)
            last_userid = r.userid
        elif (r.actionTimeDiff <= 0 and r.actionTimeDiff >= -600 or math.isnan(r.actionTimeDiff)) and r.userid == last_userid:
            seq_list.append(counter)
        else:
            counter += 1
            seq_list.append(counter)
    df['actionSeq'] = pd.Series(seq_list)
    
    time_stat = df[(df['actionSeq'] == 1) | (df['actionSeq'] == 2) | (df['actionSeq'] == 3)].groupby(['userid', 'actionSeq'])['actionTime'].agg([min, max, np.mean, np.median, np.ptp, np.std, 'count']).unstack().reset_index()
    time_stat.columns = ['userid' if i[0] == 'userid' else 'act_seq_time_stat_%s_last%d' % (i[0], i[1]) for i in time_stat.columns]
    
    time_stat.to_csv('../data/output/feat/%s' % ('action_sequence_time_stat_last123'), index=False)


# In[ ]:

def get_action_time_diff_234_56789_last_window_feature(df, window):
    df = df.copy()

    mydf = df[['userid']].drop_duplicates().reset_index(drop=True)
    
    # 234 类型的 action 的 time 的差值
    tmp = df[df['actionType'].isin([2, 3, 4])].groupby('userid').apply(lambda x: x.sort_values('actionTime', ascending=False).head(window)).reset_index(drop=True)
    tmp['act_time_rank'] = tmp.groupby('userid')['actionTime'].rank(method = 'first', ascending=False).astype(int)
    act_time = tmp.pivot('userid', 'act_time_rank', 'actionTime')
    act_time = act_time[act_time.columns[::-1]]
    act_time_diff_234 = act_time.diff(1, axis=1)
    act_time_diff_234 = act_time_diff_234.iloc[:, 1:].reset_index()
    act_time_diff_234.columns = [i if i == 'userid' else 'act_time_diff_234(' + str(i) + '-' + str(i + 1) + ')(window_' + str(window) + ')' for i in act_time_diff_234.columns]
    
    # 56789 类型的 action 的 time 的差值
    tmp = df[df['actionType'].isin([5, 6, 7, 8, 9])].groupby('userid').apply(lambda x: x.sort_values('actionTime', ascending=False).head(window)).reset_index(drop=True)
    tmp['act_time_rank'] = tmp.groupby('userid')['actionTime'].rank(method = 'first', ascending=False).astype(int)
    act_time = tmp.pivot('userid', 'act_time_rank', 'actionTime')
    act_time = act_time[act_time.columns[::-1]]
    act_time_diff_56789 = act_time.diff(1, axis=1)
    act_time_diff_56789 = act_time_diff_56789.iloc[:, 1:].reset_index()
    act_time_diff_56789.columns = [i if i == 'userid' else 'act_time_diff_56789(' + str(i) + '-' + str(i + 1) + ')(window_' + str(window) + ')' for i in act_time_diff_56789.columns]

    mydf = pd.merge(mydf, act_time_diff_234, on='userid', how='left')
    mydf = pd.merge(mydf, act_time_diff_56789, on='userid', how='left')
    
    return mydf


# In[ ]:

def get_action_stat_last_every_type_feature(df):
    df = df.copy()

    mydf = df[['userid']].drop_duplicates().reset_index(drop=True)

    # 离最近的 123456789 的 action 的时间的统计
    for t in range(1, 10):
        tmp = df[df['actionType'] == t].groupby('userid')['actionTime'].agg([min, max, np.ptp, np.std, np.mean, np.median, 'count']).reset_index()
        tmp.columns = [i if i == 'userid' else 'act_time_%s(type_%d)' % (i, t) for i in tmp.columns]
        
        mydf = pd.merge(mydf, tmp, on='userid', how='left')

    return mydf


# In[ ]:

def get_act_ord_before_type1_stat_feature(act, oord):
    act = act.copy()
    oord = oord.copy()

    mydf = oord[['userid']].drop_duplicates().reset_index(drop=True)

    ord_time = oord[oord['orderType'] == 1].groupby('userid')['orderTime'].max().reset_index()
    act = pd.merge(act, ord_time, on='userid', how='left')

    df = act[act['actionTime'] < act['orderTime']]

    act_time_stat = df.groupby('userid')['actionTime'].agg([min, max, np.ptp, np.std, np.mean, np.median, 'count']).reset_index()
    act_time_stat.columns = [i if i == 'userid' else 'act_ord_before_type1_act_time_%s' % i for i in act_time_stat.columns]
    
    act_type_size = mydf.copy()
    for t in range(1, 10):
        tmp = df[df['actionType'] == t].groupby('userid').size().reset_index()
        tmp.columns = ['userid', 'act_ord_before_type1_act_type_size(type_%d)' % t]
        act_type_size = pd.merge(act_type_size, tmp, on='userid', how='left')

    mydf = pd.merge(mydf, act_time_stat, on='userid', how='left')
    mydf = pd.merge(mydf, act_type_size, on='userid', how='left')
    
    return mydf


# In[ ]:

def get_action_time_diff_stat_feature(df):
    df = df.copy()

    mydf = df[['userid']].drop_duplicates().reset_index(drop=True)

    df = df.sort_values(['userid', 'actionTime']).reset_index(drop=True).copy()
    df['actionTimeDiff'] = df['actionTime'].diff(1)
    df = df.groupby('userid').apply(lambda x: x.iloc[1:, :]).reset_index(drop=True)

    act_time_diff_stat = df.groupby('userid')['actionTimeDiff'].agg([min, max, np.mean, np.median, np.std, sum]).reset_index()
    act_time_diff_stat.columns = [i if i == 'userid' else 'act_time_diff_%s' % i for i in act_time_diff_stat.columns]

    mydf = pd.merge(mydf, act_time_diff_stat, on='userid', how='left')
    
    return mydf


# In[ ]:

def get_action_time_diff_stat_last_window_feature(df, window):
    df = df.copy()

    mydf = df[['userid']].drop_duplicates().reset_index(drop=True)

    df = df.sort_values(['userid', 'actionTime']).reset_index(drop=True).copy()
    df['actionTimeDiff'] = df['actionTime'].diff(1)
    df = df.groupby('userid').apply(lambda x: x.iloc[1:, :]).reset_index(drop=True)
    
    tmp = df.groupby('userid').apply(lambda x: x.iloc[:-window, :]).reset_index(drop=True)
    act_time_diff_stat = tmp.groupby('userid')['actionTimeDiff'].agg([min, max, np.mean, np.median, np.std, sum]).reset_index()
    act_time_diff_stat.columns = [i if i == 'userid' else 'act_time_diff_%s(window_%d)' % (i, window) for i in act_time_diff_stat.columns]

    mydf = pd.merge(mydf, act_time_diff_stat, on='userid', how='left')
    
    return mydf


# In[ ]:

def get_action_time_last_on_every_type_feature(df):
    df = df.copy()

    mydf = df[['userid']].drop_duplicates().reset_index(drop=True)

    df = df.sort_values(['userid', 'actionTime'], ascending=[True, False]).reset_index(drop=True).copy()
    for t in range(1, 10):
        act_time = df[df['actionType'] == t].groupby('userid').apply(lambda x: x.head(1)).reset_index(drop=True)
        act_time = act_time[['userid', 'actionTime']]
        act_time.columns = ['userid', 'act_time_last(type_%d)' % t]
        
        mydf = pd.merge(mydf, act_time, on='userid', how='left')
    
    return mydf


# In[ ]:

def get_try_feat(df):
    df = df.copy()
    
    mydf = df[['userid']].drop_duplicates().reset_index(drop=True)

    df = df.sort_values(['userid', 'actionTime'], ascending=[True, False]).reset_index(drop=True).copy()
    
    last_5 = df[df.actionType == 5].drop_duplicates(subset=['userid'])
    last_6 = df[df.actionType == 6].drop_duplicates(subset=['userid'])
    time_gap_last56 = pd.merge(last_5, last_6, on='userid', how='outer')
    time_gap_last56['time_gap_last56'] = time_gap_last56.actionTime_y - time_gap_last56.actionTime_x
    mydf = pd.merge(mydf, time_gap_last56[['userid', 'time_gap_last56']], on='userid', how='left')

    tmp = df[df['actionType'] == 5].groupby('userid').apply(lambda x: x.sort_values('actionTime', ascending=False).head(2)).reset_index(drop=True)
    tmp['act_time_rank'] = tmp.groupby('userid')['actionTime'].rank(method = 'first', ascending=False).astype(int)
    act_time = tmp.pivot('userid', 'act_time_rank', 'actionTime')
    act_time = act_time[act_time.columns[::-1]]
    act_time_diff = act_time.diff(1, axis=1)
    act_time_diff = act_time_diff.iloc[:, 1:].reset_index()
    act_time_diff.columns = [i if i == 'userid' else 'act_time_diff(%d-%d)(window_%d)(type_%d)' % (i, i+1, 2, 5) for i in act_time_diff.columns]
    mydf = pd.merge(mydf, act_time_diff, on='userid', how='left')

    last_6 = df[df.actionType == 6].drop_duplicates(subset=['userid'])
    last_7 = df[df.actionType == 7].drop_duplicates(subset=['userid'])
    time_gap_last67 = pd.merge(last_6, last_7, on='userid', how='outer')
    time_gap_last67['time_gap_last67'] = time_gap_last67.actionTime_y - time_gap_last67.actionTime_x
    mydf = pd.merge(mydf, time_gap_last67[['userid', 'time_gap_last67']], on='userid', how='left')

    df['actionDate'] = pd.to_datetime(df['actionTime'], unit='s')
    df = pd.merge(df, df.drop_duplicates(subset=['userid'])[['userid', 'actionDate']], on='userid', how='left')
    df['lastDay'] = df.actionDate_x.dt.day == df.actionDate_y.dt.day
    last_day = df[df.lastDay].groupby('userid')['lastDay'].size().reset_index()
    last_day_5 = df[df.lastDay & (df.actionType == 5)].groupby('userid')['lastDay'].size().reset_index()
    tmp = pd.merge(last_day, last_day_5, on='userid', how='left')
    tmp['last_day_rate(type_5)'] = tmp.lastDay_y / tmp.lastDay_x
    mydf = pd.merge(mydf, tmp[['userid', 'last_day_rate(type_5)']], on='userid', how='left')

    last_time = df.drop_duplicates(subset=['userid'])[['userid', 'actionTime']]
    last_time.columns = ['userid', 'last_time']
    mydf = pd.merge(mydf, last_time, on='userid', how='left')

    last_4 = df[df.actionType == 4].drop_duplicates(subset=['userid'])
    last_5 = df[df.actionType == 5].drop_duplicates(subset=['userid'])
    time_gap_last45 = pd.merge(last_4, last_5, on='userid', how='outer')
    time_gap_last45['time_gap_last45'] = time_gap_last45.actionTime_y - time_gap_last45.actionTime_x
    mydf = pd.merge(mydf, time_gap_last45[['userid', 'time_gap_last45']], on='userid', how='left')

    last_1 = df[df.actionType == 1].drop_duplicates(subset=['userid'])
    last = df.drop_duplicates(subset=['userid'])
    time_gap_last1 = pd.merge(last_1, last, on='userid', how='outer')
    time_gap_last1['time_gap_last1'] = time_gap_last1.actionTime_y - time_gap_last1.actionTime_x
    mydf = pd.merge(mydf, time_gap_last1[['userid', 'time_gap_last1']], on='userid', how='left')

    last_5 = df[df.actionType == 5].drop_duplicates(subset=['userid'])
    last = df.drop_duplicates(subset=['userid'])
    time_gap_last5 = pd.merge(last_5, last, on='userid', how='outer')
    time_gap_last5['time_gap_last5'] = time_gap_last5.actionTime_y - time_gap_last5.actionTime_x
    mydf = pd.merge(mydf, time_gap_last5[['userid', 'time_gap_last5']], on='userid', how='left')

    last_6 = df[df.actionType == 6].drop_duplicates(subset=['userid'])
    last = df.drop_duplicates(subset=['userid'])
    time_gap_last6 = pd.merge(last_6, last, on='userid', how='outer')
    time_gap_last6['time_gap_last6'] = time_gap_last6.actionTime_y - time_gap_last6.actionTime_x
    mydf = pd.merge(mydf, time_gap_last6[['userid', 'time_gap_last6']], on='userid', how='left')

    tmp = df[df.actionType.isin([5, 6])].copy()
    tmp['actionTimeDiff'] = tmp['actionTime'].diff(1)
    tmp = tmp.groupby('userid').apply(lambda x: x.iloc[1:, :]).reset_index(drop=True)
    act_time_diff_stat = tmp.groupby('userid')['actionTimeDiff'].agg([min, max, np.mean, np.median, np.std, sum]).reset_index()
    act_time_diff_stat.columns = [i if i == 'userid' else 'act_time_diff_56_%s' % i for i in act_time_diff_stat.columns]
    mydf = pd.merge(mydf, act_time_diff_stat, on='userid', how='left')
    
    return mydf


# In[ ]:

action_tr = pd.read_csv('../data/input/train/action_train.csv')  # 用户行为数据
order_future_tr = pd.read_csv('../data/input/train/orderFuture_train.csv')  # 待预测数据
order_history_tr = pd.read_csv('../data/input/train/orderHistory_train.csv')  # 用户历史订单数据
user_comment_tr = pd.read_csv('../data/input/train/userComment_train.csv')  # 用户评论数据
user_profile_tr = pd.read_csv('../data/input/train/userProfile_train.csv')  # 用户个人信息

action_te = pd.read_csv('../data/input/test/action_test.csv')
order_future_te = pd.read_csv('../data/input/test/orderFuture_test.csv')
order_history_te = pd.read_csv('../data/input/test/orderHistory_test.csv')
user_comment_te = pd.read_csv('../data/input/test/userComment_test.csv')
user_profile_te = pd.read_csv('../data/input/test/userProfile_test.csv')

action = pd.concat([action_tr, action_te], axis=0).reset_index(drop=True)
order_history = pd.concat([order_history_tr, order_history_te], axis=0).reset_index(drop=True)
user_comment = pd.concat([user_comment_tr, user_comment_te], axis=0).reset_index(drop=True)
user_profile = pd.concat([user_profile_tr, user_profile_te], axis=0).reset_index(drop=True)


# In[ ]:

user_profile_feat = get_user_profile_feature(user_profile)
user_profile_feat.to_csv('../data/output/feat/%s' % 'user_profile', index=False)


# In[ ]:

user_comment_feat = get_user_comment_feature(user_comment)
user_comment_feat.to_csv('../data/output/feat/%s' % 'user_comment', index=False)


# In[ ]:

order_history_feat = get_order_history_feature(order_history)
order_history_feat.to_csv('../data/output/feat/%s' % 'order_history', index=False)


# In[ ]:

order_history_last_w_feat = get_order_history_last_w_feature(order_history)
order_history_last_w_feat.to_csv('../data/output/feat/%s' % 'order_history_last_w', index=False)


# In[ ]:

action_type_feat = get_action_type_feature(action)
action_type_feat.to_csv('../data/output/feat/%s' % 'action_type', index=False)


# In[ ]:

action_type_based_on_time_feat = get_action_type_based_on_time_feature(action)
action_type_based_on_time_feat.to_csv('../data/output/feat/%s' % 'action_type_based_on_time', index=False)


# In[ ]:

for window in [3,4,5,6,7]:
    util.log(window)
    action_type_based_on_time_last_window_feat = get_action_type_based_on_time_last_window_feature(action, window)
    action_type_based_on_time_last_window_feat.to_csv('../data/output/feat/%s%d' % ('action_type_based_on_time_last_window', window), index=False)


# In[ ]:

for window in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20]:
    util.log(window)
    action_type_num_based_on_time_last_window_feat = get_action_type_num_based_on_time_last_window_feature(action, window)
    action_type_num_based_on_time_last_window_feat.to_csv('../data/output/feat/%s%d' % ('action_type_num_based_on_time_last_window', window), index=False)


# In[ ]:

for window in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20]:
    util.log(window)
    action_type_rate_based_on_time_last_window_feat = get_action_type_rate_based_on_time_last_window_feature(action, window)
    action_type_rate_based_on_time_last_window_feat.to_csv('../data/output/feat/%s%d' % ('action_type_rate_based_on_time_last_window', window), index=False)


# In[ ]:

for window in [6]:
    util.log(window)
    action_type_row_stat_based_on_time_last_window_feat = get_action_type_row_stat_based_on_time_last_window_feature(action, window)
    action_type_row_stat_based_on_time_last_window_feat.to_csv('../data/output/feat/%s%d' % ('action_type_row_stat_based_on_time_last_window', window), index=False)


# In[ ]:

for window in [4, 7, 13, 17, 20, 25, 30]:
    util.log(window)
    action_num_based_on_time_last_window_feat = get_action_num_based_on_time_last_window_feature(action, window)
    action_num_based_on_time_last_window_feat.to_csv('../data/output/feat/%s%d' % ('action_num_based_on_time_last_window', window), index=False)


# In[ ]:

for window in [4, 7, 13, 17, 20, 25, 30]:
    util.log(window)
    action_type_num_based_on_time_last_window_feat = get_action_type_num_based_on_time_last_window_feature(action, window)
    action_type_num_based_on_time_last_window_feat.to_csv('../data/output/feat/%s%d' % ('action_type_num_based_on_time_last_window', window), index=False)


# In[ ]:

action_time_based_on_time_feat = get_action_time_based_on_time_feature(action)
action_time_based_on_time_feat.to_csv('../data/output/feat/%s' % 'action_time_based_on_time', index=False)


# In[ ]:

for window in [6]:
    util.log(window)
    action_time_based_on_time_last_window_feat = get_action_time_based_on_time_last_window_feature(action, window)
    action_time_based_on_time_last_window_feat.to_csv('../data/output/feat/%s%d' % ('action_time_based_on_time_last_window', window), index=False)


# In[ ]:

for window in [3, 6, 10, 14]:
    util.log(window)
    action_time_row_stat_based_on_time_last_window_feat = get_action_time_row_stat_based_on_time_last_window_feature(action, window)
    action_time_row_stat_based_on_time_last_window_feat.to_csv('../data/output/feat/%s%d' % ('action_time_row_stat_based_on_time_last_window', window), index=False)


# In[ ]:

for window in [3, 4, 5, 6, 7, 8]:
    util.log(window)
    action_time_diff2_based_on_time_last_window_feat = get_action_time_diff2_based_on_time_last_window_feature(action, window)
    action_time_diff2_based_on_time_last_window_feat.to_csv('../data/output/feat/%s%d' % ('action_time_diff2_based_on_time_last_window', window), index=False)


# In[ ]:

for ttype in [1,5,6,7,8,9]:
    for window in [6]:
        util.log('type=%d window=%d' % (ttype, window))
        action_time_based_on_time_last_window_on_type_feat = get_action_time_based_on_time_last_window_on_type_feature(action, window, ttype)
        action_time_based_on_time_last_window_on_type_feat.to_csv('../data/output/feat/%s%d%s%d' % ('action_time_based_on_time_last_window', window, '_on_type', ttype), index=False)


# In[ ]:

for window in [3, 4, 5, 6, 7, 8, 9, 10]:
    util.log(window)
    action_time_2order_based_on_time_last_window_feat = get_action_time_2order_based_on_time_last_window_feature(action, window)
    action_time_2order_based_on_time_last_window_feat.to_csv('../data/output/feat/%s%d' % ('action_time_2order_based_on_time_last_window', window), index=False)


# In[ ]:

for ttype in [1,5,6,7,8,9]:
    for window in [4, 7, 10]:
        util.log('type=%d window=%d' % (ttype, window))
        action_real_time_based_on_time_last_window_on_type_feat = get_action_real_time_based_on_time_last_window_on_type_feature(action, window, ttype)
        action_real_time_based_on_time_last_window_on_type_feat.to_csv('../data/output/feat/%s%d%s%d' % ('action_real_time_based_on_time_last_window', window, '_on_type', ttype), index=False)


# In[ ]:

act_ord_time_diff_feat = get_act_ord_time_diff_feature(action, order_history)
act_ord_time_diff_feat.to_csv('../data/output/feat/%s' % 'action_order_time_diff', index=False)


# In[ ]:

order_last_order_ydm_feat = get_order_last_order_ydm_feature(order_history)
order_last_order_ydm_feat.to_csv('../data/output/feat/%s' % 'order_last_order_ydm', index=False)


# In[ ]:

order_type1_ydm_feat = get_order_type1_ydm_feature(order_history)
order_type1_ydm_feat.to_csv('../data/output/feat/%s' % 'order_type1_ydm', index=False)


# In[ ]:

for window in [7,8,10,11]:
    util.log(window)
    act_ord_act_time_diff_last_window_feat = get_act_ord_act_time_diff_last_window_feature(action, order_history, window)
    act_ord_act_time_diff_last_window_feat.to_csv('../data/output/feat/%s%d' % ('act_ord_act_time_diff_last_window', window), index=False)


# In[ ]:

for window in [2,4]:
    util.log(window)
    act_ord_type1_act_time_diff_last_window_feat = get_act_ord_type1_act_time_diff_last_window_feature(action, order_history, window)
    act_ord_type1_act_time_diff_last_window_feat.to_csv('../data/output/feat/%s%d' % ('act_ord_type1_act_time_diff_last_window', window), index=False)


# In[ ]:

get_action_sequence_time_diff_feature(action)


# In[ ]:

get_action_sequence_time_stat_feature(action)


# In[ ]:

for window in [6]:
    util.log(window)
    action_time_diff_234_56789_last_window_feat = get_action_time_diff_234_56789_last_window_feature(action, window)
    action_time_diff_234_56789_last_window_feat.to_csv('../data/output/feat/%s%d' % ('action_time_diff_234_56789_last_window', window), index=False)


# In[ ]:

action_stat_last_every_type_feat = get_action_stat_last_every_type_feature(action)
action_stat_last_every_type_feat.to_csv('../data/output/feat/%s' % 'action_stat_last_every_type', index=False)


# In[ ]:

act_ord_before_type1_stat_feat = get_act_ord_before_type1_stat_feature(action, order_history)
act_ord_before_type1_stat_feat.to_csv('../data/output/feat/%s' % 'act_ord_before_type1_stat', index=False)


# In[ ]:

action_time_diff_stat_feat = get_action_time_diff_stat_feature(action)  # untest
action_time_diff_stat_feat.to_csv('../data/output/feat/%s' % 'action_time_diff_stat', index=False)


# In[ ]:

for window in [3, 4, 5, 6, 7, 8, 9]:
    util.log(window)
    action_time_diff_stat_last_window_feat = get_action_time_diff_stat_last_window_feature(action, window)
    action_time_diff_stat_last_window_feat.to_csv('../data/output/feat/%s%d' % ('action_time_diff_stat_last_window', window), index=False)


# In[ ]:

action_time_last_on_every_type_feat = get_action_time_last_on_every_type_feature(action)
action_time_last_on_every_type_feat.to_csv('../data/output/feat/%s' % 'action_time_last_on_every_type', index=False)


# In[ ]:

try_feat = get_try_feat(action)
try_feat.to_csv('../data/output/feat/%s' % 'try', index=False)

