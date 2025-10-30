# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:54:14 2025

@author: Administrator
"""

import numpy as np 
from sklearn.feature_selection import SelectKBest 
from minepy import MINE 
from sklearn import datasets 
Iris=datasets.load_iris() 
#定义求取最大信息系数的函数
def Mic(X, Y): 
 M = MINE() 
 M.compute_score(X, Y) 
 return (M.mic(), 0.5)
#构建互信息的特征提取模型
Model = SelectKBest(lambda X, Y:np.array(list(map(lambda X: Mic(X, Y), X.T))).T[0], k=2) 
#选择特征
Model.fit_transform(Iris.data, Iris.target) 
print('互信息系数：',Model.scores_) #输出互信息系数
#输出所选特征的值
print('所选特征的值为：\n',Model.fit_transform(Iris.data, Iris.target))