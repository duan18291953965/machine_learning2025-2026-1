# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:30:26 2024

@author: Administrator
"""

import matplotlib.pyplot as plt #导入绘图库
from sklearn.naive_bayes import GaussianNB #导入高斯朴素贝叶斯库
from sklearn.datasets import load_wine #导入红酒数据
from sklearn.model_selection import train_test_split #导入样本集划分模块
#加载数据
wine=load_wine()
x=wine.data
y=wine.target
#划分数据集
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
#构建模型
nb= GaussianNB()
#训练模型
nb.fit(x_train,y_train)
#输出预测精度
print("预测精度:",nb.score(x_test,y_test))
#将训练样本与测试样本分类结果可视化（方形:训练样本,圆形:测试样本）
plt.figure()
plt.scatter(x_train[:,0],x_train[:,1],c=y_train,cmap=plt.cm.cool,edgecolors='k')
plt.scatter(x_test[:,0],x_test[:,1],c=y_test,cmap=plt.cm.cool,marker='s',edgecolor='k') 
# 添加横坐标标签和纵坐标标签
plt.xlabel(wine.feature_names[0]) # 假设第一个特征是横坐标标签
plt.ylabel(wine.feature_names[1]) # 假设第二个特征是纵坐标标签
# 添加图例
plt.legend()
plt.show()
