# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:56:15 2025

@author: Administrator
"""

from sklearn.svm import LinearSVC #导入支持向量机库
from sklearn.feature_selection import SelectFromModel #导入特征选择库
#加载鸢尾花数据集
from sklearn.datasets import load_iris 
Iris = load_iris() 
X, Y = Iris.data, Iris.target 
#构建线性分类支持向量机模型并训练
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, Y) 
#使用线性分类支持向量机模型进行特征选择
Model = SelectFromModel(lsvc, prefit=True) 
X_new = Model.transform(X) 
#输出所选特征数
print(X_new.shape)