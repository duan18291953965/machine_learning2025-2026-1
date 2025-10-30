# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:54:03 2024

@author: Administrator
"""

#导入科学计算库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #导入绘图库
from sklearn .metrics import mean_squared_error #导入均方误差模块
from sklearn.model_selection import train_test_split #导入数据划分模块
from sklearn.linear_model import LinearRegression #导入线性回归库
from sklearn import preprocessing #导入数据预处理模块
#加载波士顿房价的数据集
data_url = "http://lib.stat.cmu.edu/datasets/boston"#数据网址
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)#读取数据
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
boston_df= pd.DataFrame(data,columns=['CRIM', 'ZN','INDUS','CHAS', 'NOX', 'RM', 'AGE', 'DIS','RAD','TAX','PTRADIO', 'B', 'LSTAT'])
boston_y=pd.DataFrame(target,columns=['MEDV'])#将数据转化成DataFrame格式
boston_df['MEDV']=boston_y #新增一列
boston_df.head() #显示数据前5行
boston_df.describe() #查看数据的描述信息
boston_df.corr()['MEDV'] #计算每一个特征和房价的相关系数
#显示相关系数
plt.figure(facecolor='white')#背景颜色
corr = boston_df.corr()
corr = corr['MEDV']
corr[abs(corr) > 0.5].sort_values().plot.bar() #相关系数绝对值大于0.5的特征（排序并绘制直方图）
plt.ylabel('Correlation')
plt.grid(True)
plt.show()
#数据处理
boston_df = boston_df[['LSTAT', 'PTRADIO', 'RM', 'MEDV']]#选择特征与目标值
y = np.array(boston_df['MEDV'])#目标值
boston_df = boston_df.drop(['MEDV'], axis=1)#移除目标值
X = np.array(boston_df)#特征值
#划分测试集与训练集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
#归一化
min_max_scaler = preprocessing.MinMaxScaler()
#分别对训练和测试数据的特征以及目标值进行标准化处理
X_train = min_max_scaler.fit_transform(X_train)
y_train = min_max_scaler.fit_transform(y_train.reshape(-1,1)) # reshape(-1,1)指将它转化为1列
X_test = min_max_scaler.fit_transform(X_test)
y_test = min_max_scaler.fit_transform(y_test.reshape(-1,1))
#线性回归
lr = LinearRegression()#建立线性回归模型
lr.fit(X_train, y_train)#使用训练数据进行参数估计
y_test_pred = lr.predict(X_test)#使用测试数据进行回归预测
#利用测试样本测试模型的精度
acc=lr.score(X_test,y_test)
#计算均方误差
mse=mean_squared_error(y_test,y_test_pred)
#输出模型精度和均方误差
print('精度:',acc)
print('均方误差:',mse)
#考察单个特征并进行可视化
col=X.shape[1]
for i in range(col):#遍历每一列
    plt.figure()
    linear_model=LinearRegression()#构建线性回归模型
    linear_model.fit(X_train[:,i].reshape(-1,1),y_train)#利用训练样本训练模型
    acc=linear_model.score(X_test[:,i].reshape(-1,1),y_test)#利用测试样本测试模型精度
    plt.title('Accuracy:'+str(acc))
    plt.scatter(X_train[:,i],y_train, s=30, c='green', edgecolor='black')#绘制数据点
    k=linear_model.coef_#斜率
    b=linear_model.intercept_#截距
    x=np.linspace(X_test[:,i].min(), X_test[:,i].max(),100)#根据横坐标范围生成100个数据点
    y=(k*x+b).flat#将y展平
    #绘制直线
    plt.plot(x,y,c='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
