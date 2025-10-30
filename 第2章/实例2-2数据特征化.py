# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:43:57 2025

@author: Administrator
"""

from sklearn.preprocessing import StandardScaler 
import numpy as np 
#创建一组特征数据，每一行表示一个样本，每一列表示一个特征
X = np.array([[ 4., 6., 3.], 
 [ 6., -4., 7.], 
 [ 1., 3., -8.]]) 
#定义 StandardScaler 对象
Scaler = StandardScaler () 
#求取相关变换或参数
Scaler = Scaler.fit(X) #本质是生成均值与标准差
Result = Scaler.transform(X) #根据变换或参数对数据进行处理
print("数据标准化处理后的数据：")
print(Result) 
#查看均值
print("数据均值是：")
print(Result.mean(axis=0)) #axis=1 表示对每行操作，axis=0 表示对每列操作
#查看标准差
print("数据标准差是：")
print(Result.std(axis=0))