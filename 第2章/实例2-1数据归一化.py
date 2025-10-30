# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:43:16 2025

@author: Administrator
"""

from sklearn.preprocessing import MinMaxScaler #导入归一化库
import numpy as np #导入科学计算库
#创建二维数组 X 
X = np.array([[ 4., 6., 3.], 
 [ 6., -4., 7.], 
 [ 1., 3., -8.]]) 
#定义 MinMaxScaler 对象
Scaler = MinMaxScaler() 
#求取相关变换或参数
Scaler = Scaler.fit(X) #本质是生成 min(x)和 max(x) 
Result = Scaler.transform(X) #根据变换或参数对数据进行处理
print(Result) 
Result_= Scaler.fit_transform(X) #同时执行变换或参数的求取与应用
print(Result_)