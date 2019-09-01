from sklearn.datasets import load_iris
import xgboost as xgb
from xgboost import plot_importance#用于输出特征重要性排序图
from matplotlib import pyplot as plt#画图
from sklearn.model_selection import train_test_split#切分数据为训练集和测试集

# read in the iris data
iris = load_iris()#加载数据

X = iris.data#特征
y = iris.target#标签

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234565)#切分数据

params = {   #参数配置
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': 3,
    'gamma': 0.1,
    'max_depth': 6,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
}
# params = {
#     'booster': 'gbtree',
#     'objective': 'multi:softmax',  # 多分类的问题
#     'num_class': 10,               # 类别数，与 multisoftmax 并用
#     'gamma': 0.1,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
#     'max_depth': 12,               # 构建树的深度，越大越容易过拟合
#     'lambda': 2,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
#     'subsample': 0.7,              # 随机采样训练样本
#     'colsample_bytree': 0.7,       # 生成树时进行的列采样
#     'min_child_weight': 3,
#     'silent': 1,                   # 设置成1则没有运行信息输出，最好是设置为0.
#     'eta': 0.007,                  # 如同学习率
#     'seed': 1000,
#     'nthread': 4,                  # cpu 线程数
# }

plst = params.items()#列表返回可遍历的(键, 值) 元组数组。[(key1, value1), ()]


dtrain = xgb.DMatrix(X_train, y_train)
num_rounds = 500
model = xgb.train(plst, dtrain, num_rounds)

# 对测试集进行预测, X_test类型可以是二维List，也可以是numpy的数组
dtest = xgb.DMatrix(X_test)
ans = model.predict(dtest)

# 计算准确率
cnt1 = 0
cnt2 = 0
for i in range(len(y_test)):
    if ans[i] == y_test[i]:
        cnt1 += 1
    else:
        cnt2 += 1

print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))

# 显示重要特征
plot_importance(model)

plt.show()
