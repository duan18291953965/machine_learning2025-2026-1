# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:52:26 2025

@author: Administrator
"""

from sklearn.datasets import make_regression #导入用于构造回归分析数据的库
from scipy.stats import pearsonr #导入皮尔逊相关系数库
#产生样本
X,Y = make_regression(n_samples=1000, n_features=3, n_informative=1, noise=100, 
random_state=9527) 
#分别计算每个特征与分类标记的相关系数
P1 = pearsonr(X[:,0],Y) 
P2 = pearsonr(X[:,1],Y) 
P3 = pearsonr(X[:,2],Y) 
#输出相关系数
print(P1) 
print(P2) 
print(P3)