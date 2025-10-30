# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:50:26 2025

@author: Administrator
"""

from sklearn.feature_selection import VarianceThreshold 
#定义列表 X 以保存表 2-3 中的特征值
X=[[0,0,1], [0,1,0], [1,0,0], [0,1,1], [0,1,0], [0,1,1]] 
#选择特征值为 0 或 1 且比例超过 80%的特征(布尔特征相应变量 X 的方差为 Var(X)=p*(1-p)) 
Sel=VarianceThreshold(threshold=(.8*(1-.8))) 
Y=Sel.fit_transform(X) 
#显示所选择的特征
print(Y)