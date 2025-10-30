# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:55:21 2025

@author: Administrator
"""

from sklearn.feature_selection import RFE #导入递归特征消除库
from sklearn.linear_model import LogisticRegression #导入 Logistic 回归库
#加载鸢尾花数据集
from sklearn import datasets 
Iris=datasets.load_iris() 
Name=Iris["feature_names"] 
#构建递归特征消除对象并进行特征的选择
Selector=RFE(estimator=LogisticRegression(),n_features_to_select=2).fit(Iris.data, 
Iris.target) 
#查看特征选取情况
print(Selector.support_) 
#查看特征的重要性
print(Selector.ranking_) 
#所选择的特征数
print(Selector.n_features_) 
#特征重要性排序
print("Features sorted by their rank:") 
print(sorted(zip(map(lambda X:round(X,4),Selector.ranking_),Name)))