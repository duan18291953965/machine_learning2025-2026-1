# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:47:29 2024

@author: Administrator
"""


import numpy as np #导入科学计算库
import matplotlib.pyplot as plt #导入绘图库
from sklearn.model_selection import train_test_split #导入数据划分模块
from sklearn.linear_model import LinearRegression #导入线性回归模块
#加载糖尿病数据
from sklearn.datasets import load_diabetes 
Data_diabetes = load_diabetes() 
X = Data_diabetes['data'] 
Y = Data_diabetes['target'] 
#将样本集划分为训练样本与测试样本
train_X,test_X,train_Y,test_Y = train_test_split(X,Y,train_size =0.8) 
#构建线性回归模型
linear_model = LinearRegression() 
#利用训练样本训练模型或求取模型参数
linear_model.fit(train_X,train_Y) 
#利用测试样本测试模型的精度
acc = linear_model.score(test_X,test_Y) 
print(acc) 
#考察单个特征并进行可视化
col = X.shape[1] 
for i in range(col): #遍历每 1 列
   plt.figure() 
   linear_model = LinearRegression() #构建线性回归模型
   linear_model.fit(train_X[:,i].reshape(-1,1),train_Y) #利用训练样本训练模型或求取模型参数
   acc = linear_model.score(test_X[:,i].reshape(-1,1),test_Y) #利用测试样本测试模型的精度
   plt.scatter(train_X[:,i],train_Y) #绘制数据点
    #求取相应的直线
   k =linear_model.coef_ #斜率
   b =linear_model.intercept_ #截距
   x = np.linspace(train_X[:,i].min(),train_X[:,i].max(),100) #根据横坐标范围生成100 个数据点
   y = k * x + b 
    #绘制直线
   plt.plot(x,y,c='red') 
   #显示特征列数与相应的精度
   plt.title(str(i) + ':' + str(acc)) 
   plt.xlabel('x') 
   plt.ylabel('y') 
   plt.show()