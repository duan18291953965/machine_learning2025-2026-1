# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:57:45 2025

@author: Administrator
"""

import matplotlib.pyplot as plt #导入绘图库
from sklearn.decomposition import PCA #导入主成分分析库
#加载鸢尾花数据
from sklearn.datasets import load_iris 
Iris = load_iris() 
X = Iris.data 
y = Iris.target 
#查看原特征维度
print(X.shape) 
pca = PCA(n_components=3) #实例化主成分分析对象（指定变换后的特征维度）
pca.fit(X) #求取变换矩阵
X_new = pca.transform(X) #特征变换
print(X_new) #输出变换后的样本
plt.figure() 
plt.scatter(X_new[y==0, 0], X_new[y==0, 1], c="red", label=Iris.target_names[0]) 
plt.scatter(X_new[y==1, 0], X_new[y==1, 1], c="black", label=Iris.target_names[1]) 
plt.scatter(X_new[y==2, 0], X_new[y==2, 1], c="orange", label=Iris.target_names[2]) 
plt.xlabel('x1_pca') 
plt.ylabel('x2_pca') 
plt.legend() 
plt.show()

#查看降维后新特征所带的信息量大小
print(pca.explained_variance_) 
#查看可解释性方差贡献率
print(pca.explained_variance_ratio_) 
#获取可解释性方差贡献率之和以衡量新特征所保留原特征信息的比例
print(pca.explained_variance_ratio_.sum())