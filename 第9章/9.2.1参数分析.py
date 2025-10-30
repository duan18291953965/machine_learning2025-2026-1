# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:04:58 2024

@author: Administrator
"""

import numpy as np #导入科学计算库
import matplotlib.pyplot as plt #导入绘图库
from sklearn.datasets import make_blobs #导入make_blobs数据库
from sklearn.cluster import KMeans #导入K均值聚类模块
from sklearn import metrics #导入度量模块
from scipy.spatial.distance import cdist #导入距离计算模块
#构造make_blobs样本
x, y=make_blobs(n_samples=500, n_features=2, centers=[[-1,-1], [0,0], [1,-1], [2,2]], cluster_std=[0.4, 0.3, 0.4, 0.5]) 
#利用肘部法则确定最优K值
K = range(1,10) #设置K值序列
m_cost = [] #保存代价值
#求不同K值时的代价值
for i in K:
    km = KMeans(n_clusters=i)
    km.fit(x)
    m_cost.append(sum(np.min(cdist(x,km.cluster_centers_,'euclidean'),axis=1))/x.shape[0])
#显示代价值随K值的变化曲线
plt.figure()
plt.plot(K,m_cost,'ro-')
plt.xlabel('K')
plt.ylabel('Cost')
plt.grid(True)
plt.show()
#利用非最优K值进行聚类
km = KMeans(n_clusters=3)
km.fit(x)
#查看相关度量值
y_pred = km.predict(x)
print('聚合度(K=3):',km.inertia_) #类内聚合度
silhouette = metrics.silhouette_score(x, y_pred, metric='euclidean') #轮廓系数
print('轮廓系数(K=3):',silhouette)
calinski_harabasz = metrics.calinski_harabasz_score(x,y_pred) #Calinski-Harabasz值
print('Calinski-Harabasz值(K=3):',calinski_harabasz)
#利用最优K值进行聚类
km = KMeans(n_clusters=4)
km.fit(x)
#查看相关度量值
y_pred = km.predict(x)
print('聚合度(K=4):',km.inertia_) #类内聚合度
silhouette = metrics.silhouette_score(x, y_pred, metric='euclidean') #轮廓系数
print('轮廓系数(K=4):',silhouette)
calinski_harabasz = metrics.calinski_harabasz_score(x,y_pred) #Calinski-Harabasz值
print('Calinski-Harabasz值(K=4):',calinski_harabasz)
#绘制最优K值时聚类效果图
centroids=km.cluster_centers_
plt.figure()
plt.scatter(x[y_pred==0,0], x[y_pred==0,1], s=50, c='r',linewidths=1,edgecolors='k', label='Class1')
plt.scatter(x[y_pred==1,0], x[y_pred==1,1], s=50, c='m',linewidths=1,edgecolors='k', label='Class2')
plt.scatter(x[y_pred==2,0], x[y_pred==2,1], s=50, c='g',linewidths=1,edgecolors='k', label='Class3')
plt.scatter(x[y_pred==3,0], x[y_pred==3,1], s=50, c='y',linewidths=1,edgecolors='k', label='Class4')  
plt.scatter(centroids[:,0],centroids[:,1],marker='s',s=50,c='black')#聚类中心
plt.legend(loc='best')
plt.grid(True)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
