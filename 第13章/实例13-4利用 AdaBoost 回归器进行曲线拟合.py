# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:40:25 2024

@author: Administrator
"""

from sklearn.ensemble import AdaBoostRegressor #导入 AdaBoost 回归模块
import matplotlib.pyplot as plt #导入绘图库
import numpy as np #导入科学计算库
from sklearn.metrics import r2_score #导入 R2 分数模块
#构造数据
x = np.linspace(0, 10, 200) 
x = x.reshape(-1, 1) 
y = np.sin(x).ravel() + np.random.normal(0, 0.1, x.shape[0]) 
#测试不同数量基学习器中 AdaBoost 回归器的拟合效果
AB_R2=[] 
for i in range (10): 
    AB = AdaBoostRegressor(n_estimators=(i+1)*10, random_state=0) 
    AB.fit(x, y) 
    AB_R2.append(r2_score(y,AB.predict(x))) 
#显示结果
plt.figure(1) 
plt.xlabel('Base learner') 
plt.ylabel('R2_score') 
x_range = (np.arange(10)+1)*10 
plt.plot(x_range, AB_R2, color='r',ls='--',marker='o') 
for a,b in zip(x_range,AB_R2): 
    plt.text(a+1,b,'%.2f'%b) 
plt.grid(True) 
plt.show() 
#集成不同数量个体回归器的拟合效果
#1 个个体学习器的集成
AB_1 = AdaBoostRegressor(n_estimators=1, random_state=0) 
AB_1.fit(x,y) 
y_1 = AB_1.predict(x) 
#100 个个体学习器的集成
AB_2 = AdaBoostRegressor(n_estimators=100, random_state=0) 
AB_2.fit(x,y) 
y_2 = AB_2.predict(x) 
#显示拟合结果
plt.figure(2) 
plt.scatter(x, y, color='r', label="data_point") 
plt.plot(x, y_1, color='b', label="n_estimators=1", linewidth=2) 
plt.plot(x, y_2, color='g', label="n_estimators=100", linewidth=2) 
plt.xlabel("x") 
plt.ylabel("y") 
plt.legend() 
plt.grid(True) 
plt.show()