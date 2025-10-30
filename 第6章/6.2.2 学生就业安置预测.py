# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:36:12 2025

@author: Administrator
"""

#导入科学计算相关库
import pandas as pd 
import numpy as np 
#导入绘图库
import matplotlib.pyplot as plt 
import matplotlib as mpl 
from sklearn.model_selection import cross_val_score #导入交叉验证库
from sklearn.neighbors import KNeighborsClassifier #导入 K 近邻分类库
from sklearn import preprocessing #导入数据预处理库
from sklearn.model_selection import train_test_split #导入数据划分模型
#加载数据
data = pd.read_csv('D:\\机器学习原理与应用教材课件和部分源码 - 副本\\源码\\第6章\\placement.csv',encoding=u'gbk') 
x = data.drop('placement',axis=1) 
y = data['placement'] 
print('样本数与特征数',x.shape) 
#归一化
min_max_scaler = preprocessing.MinMaxScaler() 
x = min_max_scaler.fit_transform(x) 
x_train,x_test,y_train,y_test=train_test_split(x, y,test_size=0.4) #训练数据与测试数据划分
#设置 K 列表
K = range(1,10) 
K_Acc = [] 
#求取不同 K 值交叉验证平均精度
for k in K: 
    KNN = KNeighborsClassifier(n_neighbors=k,weights='distance') 
    Acc = cross_val_score(KNN, x, y, cv=3, scoring='accuracy') 
    K_Acc.append(Acc.mean()) 
#显示不同 K 值交叉验证精度
plt.figure(1) 
plt.plot(K, K_Acc,'r-',marker='o') 
plt.xlabel('K') 
plt.ylabel('Accuracy') 
plt.grid(True) 
plt.show() 
#求取最优 K 值
k_opt = np.argmax(K_Acc)+1 
#构建 K 近邻分类器
KNN = KNeighborsClassifier(n_neighbors=k_opt) 
#训练 K 近邻分类器
KNN.fit(x_train, y_train) 
#输出预测精度
print('预测精度:',format(KNN.score(x_test,y_test),'.2f')) 
# 特征相关性分析
# 计算皮尔逊相关系数
F_1_2 = np.corrcoef(x[:,0], x[:,1])[0, 1] 
print('F1 与 F2 之间的相关性:',F_1_2) 
#分类结果可视化
x_min, x_max = x[:,0].min()-0.1, x[:,0].max()+0.1 #求第 1 个特征的最小值与最大值
y_min, y_max = x[:,1].min()-0.1, x[:,1].max()+0.1 #求第 2 个特征的最小值与最大值
#生成网格采样点并预测相应的类别
xx,yy = np.meshgrid(np.linspace(x_min, x_max, 200),np.linspace(y_min, y_max, 200)) 
grid_test = np.stack((xx.flat, yy.flat), axis=1) 
z = KNN.predict(grid_test) 
#设置前景与背景颜色
cm_pt = mpl.colors.ListedColormap(['w', 'r']) #样本点颜色
cm_bg = mpl.colors.ListedColormap(['y','g']) #背景颜色
#显示分类结果
plt.figure(2) 
plt.xlim(x_min, x_max) 
plt.ylim(y_min, y_max) 
#显示分类预测类别之间的边界
plt.pcolormesh(xx, yy, z.reshape(xx.shape), cmap=cm_bg) 
#显示测试样本真实类别
plt.scatter(x_test[:,0], x_test[:,1], c=y_test, cmap=cm_pt, marker='o',edgecolors='k') 
plt.xlabel('F1') 
plt.ylabel('F2') 
plt.show()