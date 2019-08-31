import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline

plt.style.use("bmh") # 定义图形风格bmh、ggplot、dark_background、fivethirtyeight和grayscale
plt.rc('font', family='SimHei', size=13) # 对全图文字进行修改， 解决中文乱码问题

"""
本代码参考整理自datawhale的文章《从入门到冠军 中国移动人群画像赛TOP1经验分享》
https://mp.weixin.qq.com/s/4Lt2zwbcsK9rzKGbMmHcYw
由于缺少测试标签，未尝试模型调参和修改
"""
import math
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error # 混淆矩阵、均方差、平均绝对误差
#k组用于交叉验证，StratifiedKFold：分层采样，确保训练集、测试集中各类别样本的比例与原始数据集中相同，
#cross_val_score用于交叉验证，GridSearchCV：网格搜索+交叉验证， RandomizedSearchCV随机搜索+交叉验证
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV


""" 导入数据 """
data_path = './'

test_data = pd.read_csv(data_path + 'test_dataset.csv') # 测试集
train_data = pd.read_csv(data_path + 'train_dataset.csv') # 训练集

df_data = pd.concat([train_data, test_data], ignore_index=True) # 合并，就是将测试集追加在训练集下面，增加了行数

""" 
为什么只取消一个特征的拖尾，其它特征拖尾为什么保留，即使线下提高分数也要
保留，这是因为在线下中比如逛商场拖尾的数据真实场景下可能为保安，在
训练集中可能只有一个保安，所以去掉以后线下验证会提高，但是在测试集
中也存在一个保安，如果失去拖尾最终会导致测试集保安信用分精度下降 
"""

df_data.drop(df_data[df_data['当月通话交往圈人数'] > 1750].index, inplace=True) # 删除大于1750的那一行
df_data.reset_index(drop=True, inplace=True) # 重置索引为数字，drop=True会删除原来的索引，因为上面删除了一行数据，为了保证索引连续

""" 0替换np.nan，通过线下验证发现数据实际情况缺失值数量大于0值数量，np.nan能更好的还原数据真实性 """
na_list = ['用户年龄', '缴费用户最近一次缴费金额（元）', '用户近6个月平均消费值（元）','用户账单当月总费用（元）']
for na_fea in na_list:
    df_data[na_fea].replace(0, np.nan, inplace=True) # 将0 替换为nan,inplace=True表示修改原数据

""" 话费敏感度0替换，通过线下验证发现替换为众数能比np.nan更好的还原数据真实性 """
df_data['用户话费敏感度'].replace(0, df_data['用户话费敏感度'].mode()[0], inplace=True) # 众数替换0，inplace=True表示修改原数据

""" x / (y + 1) 避免无穷值Inf，采用高斯平滑 + 1 """
df_data['话费稳定'] = df_data['用户账单当月总费用（元）'] / (df_data['用户当月账户余额（元）'] + 1)  # 新特征
df_data['相比稳定'] = df_data['用户账单当月总费用（元）'] / (df_data['用户近6个月平均消费值（元）'] + 1) # 新特征
df_data['缴费稳定'] = df_data['缴费用户最近一次缴费金额（元）'] / (df_data['用户近6个月平均消费值（元）'] + 1) # 新特征

df_data['当月是否去过豪华商场'] = (df_data['当月是否逛过福州仓山万达'] + df_data['当月是否到过福州山姆会员店']).map(lambda x: 1 if x > 0 else 0) # map对指定序列做映射, reduce对参数序列中元素进行累积
df_data['应用总使用次数'] = df_data['当月网购类应用使用次数'] + df_data['当月物流快递类应用使用次数'] + df_data['当月金融理财类应用使用总次数'] + df_data['当月视频播放类应用使用次数'] + df_data['当月飞机类应用使用次数'] + df_data['当月火车类应用使用次数'] + df_data['当月旅游资讯类应用使用次数']


df_data['缴费方式'] = 0
df_data.loc[(df_data['缴费用户最近一次缴费金额（元）'] != 0) & (df_data['缴费用户最近一次缴费金额（元）'] % 10 == 0), '缴费方式'] = 1 # 交费大于0且是10的倍数，则设缴费方式1
df_data.loc[(df_data['缴费用户最近一次缴费金额（元）'] != 0) & (df_data['缴费用户最近一次缴费金额（元）'] % 10 > 0), '缴费方式'] = 2 # 交费大于0且不是10的倍数，则设缴费方式2

f, ax = plt.subplots(figsize=(20, 6)) # 画布
sns.boxplot(data=df_data, x='缴费方式', y='信用分', ax=ax) # 箱形图
plt.show()

df_data['信用资格'] = df_data['用户网龄（月）'].apply(lambda x: 1 if x > 12 else 0)

f, ax = plt.subplots(figsize=(10, 6))
sns.boxenplot(data=df_data, x='信用资格', y='信用分', ax=ax)
plt.show()


df_data['敏度占比'] = df_data['用户话费敏感度'].map({1:1, 2:3, 3:3, 4:4, 5:8}) # 使用字典进行映射

f, ax = plt.subplots(1, 2, figsize=(20, 6))

sns.boxenplot(data=df_data, x='敏度占比', y='信用分', ax=ax[0])
sns.boxenplot(data=df_data, x='用户话费敏感度', y='信用分', ax=ax[1])
plt.show()

lab = '信用分'

X = df_data.loc[df_data[lab].notnull(), (df_data.columns != lab) & (df_data.columns != '用户编码')] # 训练数据的特征

y = df_data.loc[df_data[lab].notnull()][lab] # 训练数据的目标值

X_pred = df_data.loc[df_data[lab].isnull(), (df_data.columns != lab) & (df_data.columns != '用户编码')] # 测试数据的特征，label为null的就是测试数据


--自定义评价函数
def feval_lgb(y_pred, train_data):
    y_true = train_data.get_label()

    score = 1 / (1 + mean_absolute_error(y_true, y_pred)) # 1/（1 + mae(y_true, y_pred)）

    score += 0.0001

    return 'acc_score', score, True

lgb_param_l1 = {

    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'regression_l1',
    'metric': {'l1'},
    'min_child_samples': 46,
    'min_child_weight': 0.01,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'num_leaves': 31,
    'max_depth': 5,
    'lambda_l2': 1,
    'lambda_l1': 0,
    'n_jobs': -1,
    'seed': 4590,
}

lgb_param_l2 = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'regression_l2',
    'metric': 'None',
    'feature_fraction': 0.6,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'num_leaves': 40,
    'max_depth': 7,
    'lambda_l2': 1,
    'lambda_l1': 0,
    'n_jobs': -1,
}
# 不使用自定义评价函数
# n_fold = 5
# y_counts = 0
# y_scores = np.zeros(5) #[0,0,0,0,0]
# y_pred_l1 = np.zeros([5, X_pred.shape[0]]) # 存预测结果,shape[0]取行数
# y_pred_all_l1 = np.zeros(X_pred.shape[0])
#
# for n in range(1):
#     kfold = KFold(n_splits=n_fold, shuffle=True, random_state=2019 + n) #k折交叉验证
#     kf = kfold.split(X, y) # 切分数据集
#
#     for i, (train_iloc, test_iloc) in enumerate(kf): #循环训练
#         print("{}、".format(i + 1), end='')
#         X_train, X_test, y_train, y_test = X.iloc[train_iloc, :], X.iloc[test_iloc, :], y[train_iloc], y[test_iloc]
#
#         lgb_train = lgb.Dataset(X_train, y_train)
#         lgb_valid = lgb.Dataset(X_test, y_test, reference=lgb_train)
#         lgb_model = lgb.train(train_set=lgb_train, valid_sets=lgb_valid, # 训练集、验证集
#                               params=lgb_param_l1, num_boost_round=6000, verbose_eval=-1, early_stopping_rounds=100)
#         # verbose_eval迭代多少次打印 early_stopping_rounds 多少次分数没有提高则停止
#         #         lgb_model.save_model('lgb_model.txt')
#
#         y_scores[y_counts] = lgb_model.best_score['valid_0']['l1']
#         #         y_scores[y_counts] = lgb_model.best_score[list(model.best_score_.keys())[0]]['mae']
#         #         y_scores[y_counts] = lgb_model.best_score['valid_0']['acc_score']
#         y_pred_l1[y_counts] = lgb_model.predict(X_pred, num_iteration=lgb_model.best_iteration)
#         y_pred_all_l1 += y_pred_l1[y_counts]
#         y_counts += 1
#
# y_pred_all_l1 /= y_counts #取5次预测的均值
# print(y_scores, y_scores.mean())
#
# lgb_param_l1 = {
#
#     'learning_rate': 0.01,
#     'boosting_type': 'gbdt',
#     'objective': 'regression_l1',
#     'metric': 'None',
#     'min_child_samples': 46,
#     'min_child_weight': 0.01,
#     'feature_fraction': 0.6,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 2,
#     'num_leaves': 31,
#     'max_depth': 5,
#     'lambda_l2': 1,
#     'lambda_l1': 0,
#     'n_jobs': -1,
#     'seed': 4590,
# }

# 使用自定以评价函数 l1模型
n_fold = 5
y_counts = 0
y_scores = np.zeros(5)
y_pred_l1 = np.zeros([5, X_pred.shape[0]])
y_pred_all_l1 = np.zeros(X_pred.shape[0])

for n in range(1):
    kfold = KFold(n_splits=n_fold, shuffle=True, random_state=2019 + n)
    kf = kfold.split(X, y)

    for i, (train_iloc, test_iloc) in enumerate(kf):
        print("{}、".format(i + 1), end='')
        X_train, X_test, y_train, y_test = X.iloc[train_iloc, :], X.iloc[test_iloc, :], y[train_iloc], y[test_iloc]

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_test, y_test, reference=lgb_train)
        lgb_model = lgb.train(train_set=lgb_train, valid_sets=lgb_valid, feval=feval_lgb,
                              params=lgb_param_l1, num_boost_round=6000, verbose_eval=-1, early_stopping_rounds=100)

        y_scores[y_counts] = lgb_model.best_score['valid_0']['acc_score']
        y_pred_l1[y_counts] = lgb_model.predict(X_pred, num_iteration=lgb_model.best_iteration)
        y_pred_all_l1 += y_pred_l1[y_counts]
        y_counts += 1

y_pred_all_l1 /= y_counts
print(y_scores, y_scores.mean())

#l2模型
n_fold = 5
y_counts = 0
y_scores = np.zeros(5)
y_pred_l2 = np.zeros([5, X_pred.shape[0]])
y_pred_all_l2 = np.zeros(X_pred.shape[0])

for n in range(1):
    kfold = KFold(n_splits=n_fold, shuffle=True, random_state=2019 + n)
    kf = kfold.split(X, y)

    for i, (train_iloc, test_iloc) in enumerate(kf):
        print("{}、".format(i + 1), end='')
        X_train, X_test, y_train, y_test = X.iloc[train_iloc, :], X.iloc[test_iloc, :], y[train_iloc], y[test_iloc]

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_test, y_test, reference=lgb_train)
        lgb_model = lgb.train(train_set=lgb_train, valid_sets=lgb_valid, feval=feval_lgb,
                              params=lgb_param_l2, num_boost_round=6000, verbose_eval=-1, early_stopping_rounds=100)

        y_scores[y_counts] = lgb_model.best_score['valid_0']['acc_score']
        y_pred_l2[y_counts] = lgb_model.predict(X_pred, num_iteration=lgb_model.best_iteration)
        y_pred_all_l2 += y_pred_l1[y_counts]
        y_counts += 1

y_pred_all_l2 /= y_counts
print(y_scores, y_scores.mean())

"""
 模型融合，低分段和高分段使用mae, 中分段使用mse
"""
submit = pd.DataFrame()
submit['id'] = df_data[df_data['信用分'].isnull()]['用户编码']

submit['score1'] = y_pred_all_l1
submit['score2'] = y_pred_all_l2

submit = submit.sort_values('score1')
submit['rank'] = np.arange(submit.shape[0])

min_rank = 100
max_rank = 50000 - min_rank

l1_ext_rate = 1
l2_ext_rate = 1 - l1_ext_rate
il_ext = (submit['rank'] <= min_rank) | (submit['rank'] >= max_rank)

l1_not_ext_rate = 0.5
l2_not_ext_rate = 1 - l1_not_ext_rate
il_not_ext = (submit['rank'] > min_rank) & (submit['rank'] < max_rank)

submit['score'] = 0
submit.loc[il_ext, 'score'] = (submit[il_ext]['score1'] * l1_ext_rate + submit[il_ext]['score2'] * l2_ext_rate + 1 + 0.25)
submit.loc[il_not_ext, 'score'] = submit[il_not_ext]['score1'] * l1_not_ext_rate + submit[il_not_ext]['score2'] * l2_not_ext_rate + 0.25
""" 输出文件 """
submit[['id', 'score']].to_csv('submit.csv')
