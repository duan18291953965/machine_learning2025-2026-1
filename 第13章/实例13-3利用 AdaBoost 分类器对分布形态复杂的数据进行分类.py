# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:39:27 2024

@author: Administrator
"""

from sklearn.ensemble import AdaBoostClassifier #导入 AdaBoost 分类器模块
from sklearn.tree import DecisionTreeClassifier #导入决策树模块
import matplotlib.pyplot as plt #导入绘图库
import matplotlib as mpl 
import numpy as np #导入科学计算库
from sklearn.metrics import accuracy_score 
from sklearn.datasets import make_gaussian_quantiles 
from sklearn.model_selection import train_test_split #导入数据划分模块
#构建多类别数据
x, y = make_gaussian_quantiles(mean=(1,1),cov=2.0,n_samples=1000, n_features=2, 
n_classes=3, random_state=1) 
#将数据划分为训练数据与测试数据
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3) 
#构建 AdaBoost 分类器（SAMME.R）
ABC_SR = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=300, 
learning_rate=1) 
ABC_SR.fit(x_train, y_train) 
#显示分类界线
N, M = 200,200 
x1_min, x2_min = x.min(axis=0) #求最小值
x1_max, x2_max = x.max(axis=0) #求最大值
t1 = np.linspace(x1_min, x1_max, N) #生成横坐标
t2 = np.linspace(x2_min, x2_max, M) #生成纵坐标
x1,x2 = np.meshgrid(t1,t2) #生成网格采样点
grid_test = np.stack((x1.flat, x2.flat), axis=1) #利用采样点生成样本
y_predict = ABC_SR.predict(grid_test) #预测样本类别
cm_pt = mpl.colors.ListedColormap(['w', 'g','c']) #散点颜色
cm_bg = mpl.colors.ListedColormap(['r', 'y','m']) #背景颜色
plt.figure(1) 
plt.xlim(x1_min, x1_max);plt.ylim(x2_min, x2_max) #设置坐标范围
plt.pcolormesh(x1,x2,y_predict.reshape(x1.shape), cmap=cm_bg) #绘制网格背景
plt.scatter(x[:,0],x[:,1],c=y,cmap=cm_pt,marker='o',edgecolors='k') #绘制散点
plt.xlabel('x1') 
plt.ylabel('x2') 
plt.grid(True) 
plt.show() 
#构建 AdaBoost 分类器（SAMME）
ABC_S = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=300, 
learning_rate=1.5,algorithm="SAMME") 
ABC_S.fit(x_train, y_train) 
#计算误差
ABC_SR_ERR = [] 
ABC_S_ERR = [] 
for ABC_SR_predict, ABC_S_predict in zip(ABC_SR.staged_predict(x_test), 
ABC_S.staged_predict(x_test)):
    ABC_SR_ERR.append(1.0 - accuracy_score(ABC_SR_predict, y_test)) 
    ABC_S_ERR.append(1.0 - accuracy_score(ABC_S_predict, y_test)) 
#绘制误差变化曲线
plt.figure(2) 
plt.plot(range(len(ABC_S_ERR)),ABC_S_ERR,"b",label="SAMME",alpha=0.5) 
plt.plot(range(len(ABC_SR_ERR)), ABC_SR_ERR, "r", label="SAMME.R", alpha=0.5) 
plt.ylabel('Error') 
plt.xlabel('Number of Trees') 
plt.legend(loc='best') 
plt.grid(True) 
plt.show()