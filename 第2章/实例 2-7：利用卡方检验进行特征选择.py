# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:53:03 2025

@author: Administrator
"""

from sklearn.datasets import load_iris #导入鸢尾花数据集
from sklearn.feature_selection import SelectKBest #导入特征选择库
from sklearn.feature_selection import chi2 #导入卡方检验库
#加载鸢尾花数据集
Iris = load_iris() 
X, Y = Iris.data, Iris.target 
print(X.shape) #样本数与初始特征数
#选择两个最佳特征
X_new = SelectKBest(chi2, k=2).fit_transform(X, Y)
#   删除了两个特征
print(X_new.shape) #样本数与选择的特征数