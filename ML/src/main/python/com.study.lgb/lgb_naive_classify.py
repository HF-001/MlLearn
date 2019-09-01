import numpy as np
import lightgbm as lgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score

iris=datasets.load_iris()

X_train,X_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.3)


train_data=lgb.Dataset(X_train,label=y_train)
validation_data=lgb.Dataset(X_test,label=y_test)

params={
    'learning_rate':0.1,
    'lambda_l1':0.1,
    'lambda_l2':0.2,
    'max_depth':4,
    'objective':'multiclass',
    'num_class':3,  #lightgbm.basic.LightGBMError: b‘Number of classes should be specified and greater than 1 for multiclass training‘
}

clf=lgb.train(params,train_data,valid_sets=[validation_data])

y_pred=clf.predict(X_test)
print(y_pred)
y_pred=[list(x).index(max(x)) for x in y_pred]
print(y_pred)
print(accuracy_score(y_test,y_pred))