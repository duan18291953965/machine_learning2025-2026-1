# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:03:47 2025

@author: Administrator
"""

import matplotlib.pyplot as plt #导入绘图库
#导入线性判别分析库
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
#加载鸢尾花数据
from sklearn import datasets 
Iris = datasets.load_iris() 
X = Iris.data 
y = Iris.target 
#查看样本数与特征数
print(X.shape) 
#实例化线性判别分析对象并指定新特征的维度
lda = LinearDiscriminantAnalysis(n_components=2) 
#特征变换
X_new = lda.fit(X, y).transform(X) 
#查看降维后的样本数与特征数
print(X_new.shape) 
#查看降维后特征累积可解释性方差贡献率
print(lda.explained_variance_ratio_.sum()) 
#降维后特征可视化
plt.figure() 
plt.scatter(X_new[y==0, 0], X_new[y==0, 1], c="red", label=Iris.target_names[0]) 
plt.scatter(X_new[y==1, 0], X_new[y==1, 1], c="black", label=Iris.target_names[1]) 
plt.scatter(X_new[y==2, 0], X_new[y==2, 1], c="orange", label=Iris.target_names[2]) 
plt.xlabel('x1_lda') 
plt.ylabel('x2_lda') 
plt.legend() 
plt.show()