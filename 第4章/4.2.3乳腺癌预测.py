# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 17:13:26 2024

@author: Administrator
"""
import matplotlib.pyplot as plt  #导入绘图库
import numpy as np #导入科学计算库
from sklearn.datasets import load_breast_cancer #导入数据库
from sklearn.model_selection import train_test_split #导入样本集划分模块
from sklearn import preprocessing #导入数据预处理库
from sklearn.linear_model import LogisticRegression #导入Logistic回归库
from sklearn.model_selection import cross_val_score #导入交叉验证库
from sklearn.feature_selection import SelectFromModel #导入特征选择库
#加载数据
Cancer=load_breast_cancer()
x = Cancer.data #特征值
y = Cancer.target #目标值
#输出数据基本信息
print('数据基本信息: {0}; Cancer_No: {1}; Cancer_Yes: {2}'.format(x.shape, y[y==1].shape[0], y[y==0].shape[0]))
print('特征名称:',Cancer.feature_names)
#将样本集划分为训练样本与测试样本
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)
# 对数据进行标准化处理
transfer = preprocessing.StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)
#构建逻辑回归模型 
LR= LogisticRegression()
LR.fit(x_train, y_train)
#模型评估
print('预测精度:',LR.score(x_test, y_test))
#在L2正则化的基础上通过遍历C值的方式确定特征提取前与特征提取后的精度
all_features = []
selected_features = []
C = np.arange(0.01,10,0.5)
for i in C:
    #构建Logistic回归模型
    LR = LogisticRegression (penalty='l2',solver="liblinear",C=i,random_state=100)
    #特征提取前交叉验证精度
    all_features.append(cross_val_score(LR,x_train,y_train,cv=10).mean())
    #特征提取后交叉验证精度
    X_new = SelectFromModel(LR,norm_order=1).fit(x_train,y_train)
    X_new_train = X_new.transform(x_train)
    selected_features.append(cross_val_score(LR,X_new_train,y_train,cv=10).mean())
#输出特征提取前模型精度最高值及对应的C值
print('特征提取前模型精度最高值及对应的C值:',max(all_features),C[all_features.index(max(all_features))])
#输出特征提取后模型精度最高值及对应的C值
print('特征提取后模型精度最高值及对应的C值:',max(selected_features),C[selected_features.index(max(selected_features))])
plt.figure(figsize=(10,5))
plt.plot(C,all_features,label="All Features")
plt.plot(C,selected_features,label="Selected Features")
plt.xticks(C)
plt.grid(True)
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()
#构建与训练Logistic回归模型
LR = LogisticRegression (penalty='l2',solver='liblinear',C=0.5,random_state=100)
x_new = SelectFromModel(LR,norm_order=1).fit(x_train,y_train)
x_new_train = x_new.transform(x_train)
LR.fit(x_new_train,y_train)
#利用训练数据测试Logistic回归模型的精度
print('训练数据相应的精度:',cross_val_score(LR,x_new_train,y_train,cv=10).mean())
#利用测试数据测试Logistic回归模型的精度
x_new_test = x_new.transform(x_test)
print('测试数据相应的精度:',cross_val_score(LR,x_new_test,y_test,cv=10).mean())



