# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 17:01:25 2024

@author: Administrator
"""

import numpy as np #导入科学计算库
#导入绘图库
import matplotlib.pyplot as plt 
import matplotlib as mpl
from sklearn.linear_model import LogisticRegression #导入Logistic回归库
from sklearn.datasets import make_blobs #导入make_blobs数据库
#生成数据
x, y = make_blobs(n_samples=100, n_features=2, centers=[[1,1], [2.5,3]], cluster_std=[0.8, 0.8]) 
#构建Logistic回归模型
LR = LogisticRegression()
#模型训练
LR.fit(x,y) 
#输出样本的预测概率值
print('预测精度:', LR.score(x,y))
#显示分类界线[方法1]
N, M = 1000,1000
x1_min, x2_min = x.min(axis=0) #求最小值
x1_max, x2_max = x.max(axis=0) #求最大值
t1 = np.linspace(x1_min, x1_max, N) #生成横坐标
t2 = np.linspace(x2_min, x2_max, M) #生成纵坐标
x1,x2 = np.meshgrid(t1,t2) #生成网格采样点
grid_test = np.stack((x1.flat, x2.flat), axis=1) #利用采样点生成样本
y_predict = LR.predict(grid_test) #预测样本类别
cm_pt = mpl.colors.ListedColormap(['w', 'g']) #散点颜色
cm_bg = mpl.colors.ListedColormap(['r', 'y']) #背景颜色
plt.figure()
plt.xlim(x1_min, x1_max);plt.ylim(x2_min, x2_max) #设置坐标范围
plt.pcolormesh(x1,x2,y_predict.reshape(x1.shape), cmap=cm_bg) #绘制网格背景
plt.scatter(x[:,0],x[:,1],c=y,cmap=cm_pt,marker='o',edgecolors='k') #绘制散点
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)
plt.show()
#显示分类界线[方法2]
w=LR.coef_ #回归系数
b=LR.intercept_ #截距
#可视化拟合分类结果
colors=mpl.colors.ListedColormap(['w','g'])    #设置不同类别的颜色映射
plt.scatter(x[:,0],x[:,1],c=y,cmap=colors,marker='o',edgecolors='k')   #利用散点图对可视化样本数据
#画出分类界线
x1=np.linspace(-2,5)
x2=(w[0][0]*x1+b)/(-w[0][1])   
plt.plot(x1,x2,'r-',linewidth=2)
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)
plt.show()
