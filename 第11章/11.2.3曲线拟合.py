# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:05:16 2024

@author: Administrator
"""

import numpy as np #导入科学计算库
from sklearn.neural_network import MLPRegressor #导入人工神经网络模块
import matplotlib.pyplot as plt #导入绘图库
from sklearn.metrics import r2_score #导入 R2 分数模块
# 构造数据点
x= np.c_[np.linspace(0,2*np.pi,40)] 
y = np.sin(x).ravel() + np.random.normal(0,0.2,len(x)) 
# 曲线拟合（不同神经元与不同激活函数）
R2_Acc = [] 
plt.figure(1) 
plt.scatter(x, y, color='c', label='Data Points',linewidths=1,edgecolors='k') 
NN_1 = MLPRegressor(solver='lbfgs',activation='logistic',hidden_layer_sizes=(2,2)) 
y_1 = NN_1.fit(x, y).predict(x) 
R2_Acc.append(r2_score(y, y_1)) 
plt.plot(x, y_1, c='b', lw=2, linestyle='-.',label='Activation:logistic,Layer:(2,2)') 
NN_2 = MLPRegressor(solver='lbfgs',activation='logistic',hidden_layer_sizes=(8,8)) 
y_2 = NN_2.fit(x, y).predict(x) 
R2_Acc.append(r2_score(y, y_2)) 
plt.plot(x, y_2, c='r', lw=2, linestyle='-',label='Activation:logistic,Layer:(8,8)') 
NN_3 = MLPRegressor(solver='lbfgs',activation='relu',hidden_layer_sizes=(2,2)) 
y_3 = NN_3.fit(x, y).predict(x) 
R2_Acc.append(r2_score(y, y_3)) 
plt.plot(x, y_3, c='g', lw=2, linestyle='--',label='Activation:relu,Layer:(2,2)') 
NN_4 = MLPRegressor(solver='lbfgs',activation='relu',hidden_layer_sizes=(8,8)) 
y_4 = NN_4.fit(x, y).predict(x) 
R2_Acc.append(r2_score(y, y_4)) 
plt.plot(x, y_4, c='m', lw=2, linestyle=':',label='Activation:relu,Layer:(8,8)') 
plt.grid(True) 
plt.legend() 
plt.show() 
# 显示 R2 值（不同神经元与不同激活函数）
plt.figure() 
plt.xlabel('Model') 
plt.ylabel('R2') 
model = ['Logistic(2,2)','Logistic(8,8)','ReLU(2,2)','ReLU(8,8)'] 
plt.bar(np.arange(4),R2_Acc,0.5, color=['r','g','b','c'], tick_label=model) 
for a,b in zip(np.arange(4),R2_Acc): 
    plt.text(a,b,'%.3f'%b,ha='center') 
plt.show()