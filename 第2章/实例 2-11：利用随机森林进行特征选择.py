# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:57:01 2025

@author: Administrator
"""

from sklearn.ensemble import RandomForestRegressor #导入随机森林库
from sklearn.feature_selection import SelectFromModel #导入特征选择库
#加载鸢尾花数据集
from sklearn.datasets import load_iris 
Iris = load_iris() 
X, Y = Iris.data, Iris.target 
print(X.shape) 
#构建随机森林模型并训练
Clf=RandomForestRegressor() 
Clf = Clf.fit(X, Y) 
#查看每个特征的重要性
print("特征重要性是：")
print(Clf.feature_importances_) 
#使用随机森林模型进行特征的选择
Model = SelectFromModel(Clf, prefit=True) 
X_new = Model.transform(X) 
#输出所选特征数
print(X_new.shape)