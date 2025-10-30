# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 09:00:56 2024

@author: Administrator
"""
import numpy as np #导入科学计算库
#导入绘图库
import matplotlib.pyplot as plt 
import matplotlib as mpl 
from sklearn.neighbors import KNeighborsClassifier #导入 K 近邻分类模块

#构造训练样本
x_train = np.array([
    [4,         5],
    [6,         7],
    [4.8,       7],
    [5.5,       8],
    [7,         8],
    [10,        11],
    [9,         14]])
y_train = ['A', 'A', 'A', 'A', 'B', 'B', 'B'] 
#测试样本
x_test = np.array([
    [3.5,       7],
    [9,         13],
    [8.7,       10],
    [5,         6],
    [7.5,       8],
    [9.5,       12],
    [1.5,       10],
    [8.5,       9]])
plt.figure(1) 
plt.xlabel('X');
plt.ylabel('Y')
plt.plot(x_train[0:4,0], x_train[0:4,1], color='red', marker='o', label='One Class (A)', linestyle='') 
plt.plot(x_train[4:7,0], x_train[4:7,1], color='blue', marker='s', label='Two Class (B)', linestyle='') 
plt.plot(x_test[:,0], x_test[:,1], color='green', marker='^', label='Sample', linestyle='') 
for i in range(len(x_test)): 
    plt.text(x_test[i,0]-0.3,x_test[i,1]+0.3, str(i) + '->?') 
plt.legend(loc='upper left') 
plt.grid(True) 
plt.show() 
#构建 K 近邻分类器并利用训练样本进行训练
#采用距离倒数权重时设置 weights='distance'
knn = KNeighborsClassifier(n_neighbors=3,weights='distance',algorithm='auto')
knn.fit(x_train, y_train)
#利用测试样本对模型进行测试
y_predict = knn.predict(x_test)
print(y_predict)
#分类结果可视化
x_min, x_max = min(x_train[:,0].min(),x_test[:,0].min())-1, max(x_train[:,0].max(),x_test[:,0].max())+1 #求第 1 个特征的最小值与最大值
y_min, y_max = min(x_train[:,1].min(),x_test[:,1].min())-1, max(x_train[:,1].max(),x_test[:,1].max())+1 #求第 2 个特征的最小值与最大值
#生成网格采样点
xx,yy = np.meshgrid(np.linspace(x_min, x_max, 200),np.linspace(y_min, y_max, 200))
grid_test = np.stack((xx.flat, yy.flat), axis=1) #测试点
z = knn.predict(grid_test)

z = np.array([0 if x=='A' else 1 for x in z])
#生成前景与背景颜色
cm_pt = mpl.colors.ListedColormap(['w', 'k']) #样本点颜色
cm_bg = mpl.colors.ListedColormap(['c', 'y']) #背景颜色
plt.figure(2)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.pcolormesh(xx, yy, z.reshape(xx.shape), cmap=cm_bg) #绘制网格背景
#显示训练样本
plt.plot(x_train[0:4,0], x_train[0:4,1], color='black', marker='o', label='One Class (A)',
linestyle='')
plt.plot(x_train[4:7,0], x_train[4:7,1], color='black', marker='s', label='Two Class (B)',
linestyle='')
#显示测试样本与分类结果
for i in range(len(x_test)):
    if y_predict[i] == 'A':
        plt.plot(x_test[i,0], x_test[i,1], color='black', marker='o')
        plt.text(x_test[i,0]-0.3, x_test[i,1]+0.3, str(i) + '->A')
    else:
        plt.plot(x_test[i,0], x_test[i,1], color='black', marker='s')
        plt.text(x_test[i,0]-0.3, x_test[i,1]+0.3, str(i) + '->B')
plt.legend(loc='upper left')
plt.grid(True)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()