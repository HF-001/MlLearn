import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
np.random.seed(123)

"""
线性回归：y=x*w+b
模型设定样本误差服从高斯分布，通过替换误差项，计算样本的整体误差概率（连乘），取对数变换，进而得到最小二乘法的公式，
接着，该公式对w、b求偏导数，利用梯度下降法更新参数
"""
X = 2 * np.random.rand(500, 1)
y = 5 + 3 * X + np.random.randn(500, 1)
fig = plt.figure(figsize=(8,6))
plt.scatter(X, y)
plt.title("Dataset")
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y)
print(f'Shape X_train: {X_train.shape}')
print(f'Shape y_train: {y_train.shape}')
print(f'Shape X_test: {X_test.shape}')
print(f'Shape y_test: {y_test.shape}')

class LinearRegression:#类名
    def _init_(self):#初始化
        pass#什么也不做，只是单纯的防止语句错误
    def train_gradient_descent(self,x,y,learning_rate=0.01,n_iters=100):#梯度下降法训练，x，y不用解释，
        #学习率就是学习的速度，n_iters是迭代的次数
        n_samples,n_features=x.shape#将x的大小x=500，y=1分别分给样品和特征
        self.weights=np.zeros(shape=(n_features,1))#以下都是第零步，给权值赋为0,这里的n_features==1，n_features*1
        self.bias=0#偏差赋值为0
        costs=[]#申请一个损失数组

        for i in range(n_iters):#迭代n_iters次
            y_predict=np.dot(x,self.weights)+self.bias#第一步：y_predict=X*w+b, np.dot()一维点积，二维矩阵相乘
            cost=(1/n_samples)*np.sum((y_predict-y)**2)#第二步，得训练集的损失,最小二乘法
            costs.append(cost)#将损失加到损失数组里面

            if i%100==0:#每过一百次输出一下损失
                print(f"Cost at iteration{i}:{cost}")

            dJ_dw=(2/n_samples)*np.dot(x.T,(y_predict-y))#第三步 第一个公式，得对应偏导数的梯度
            dJ_db=(2/n_samples)*np.sum((y_predict-y))#第三步 第二个公式

            self.weights=self.weights-learning_rate*dJ_dw#第四步 第一个公式，刷新权值
            self.bias=self.bias-learning_rate*dJ_db#第四步 第二个公式，刷新偏差

        return self.weights,self.bias,costs#返回所得参数
    def train_normal_equation(self,x,y):#正规的方程训练
        self.weights=np.dot(np.dot(np.linalg.inv(np.dot(x.T,x)),x.T),y)#正规方程公式
        self.bias=0

        return self.weights,self.bias
    def predict(self,x):
        return np.dot(x,self.weights)+self.bias

regressor=LinearRegression() #梯度下降法的一个实例化对象
w_trained,b_trained,costs=regressor.train_gradient_descent(X_train,y_train,learning_rate=0.005,n_iters=600)
#对该对象进行训练，并获取训练之后的权值与偏差
fig=plt.figure(figsize=(8,6))#设置画布大小
plt.plot(np.arange(600),costs)#设置绘画内容，x轴为迭代次数，y轴为训练集的损失
plt.title("Development of cost during training")#标题
plt.xlabel("Numbers of iterations: ")#x轴标题
plt.ylabel("Cost")#y轴标题
plt.show()#显示

n_samples,_=X_train.shape#这里想要的只有训练集的行数，_代表的也是一个变量名，只是为1，为什么用
#相当于被抛弃的那种。之所以写在这里，也是为了防止程序出错
n_samples_test,_=X_test.shape#这里想要的是测试集的行数

y_p_train=regressor.predict(X_train)#计算训练集中的特征与权值的线性组合，借鉴梯度下降法中的第一步
y_p_test=regressor.predict(X_test)#计算测试集中的特征与权值的线性组合
error_train=(1/n_samples)*np.sum((y_p_train-y_train)**2)#这里计算的是训练集的的误差
error_test=(1/n_samples_test)*np.sum((y_p_test-y_test)**2)#这里计算的是测试集的的误差
print(f"error on training set:{np.round(error_train,4)}")#输出训练集的误差，保留四位小数
print(f"error on testing set:{np.round(error_test)}")#输出测试集的误差

fig=plt.figure(figsize=(8,6))#设置画布大小
plt.scatter(X_train,y_train)#绘制训练集的散点图
plt.scatter(X_test,y_p_test)#绘制测试集的散点图，注意这里的y_p_test是正态之后的测试集的y
plt.xlabel("First feature")#x轴的标题
plt.ylabel("Second feature")#y轴的标题
plt.show()#显示