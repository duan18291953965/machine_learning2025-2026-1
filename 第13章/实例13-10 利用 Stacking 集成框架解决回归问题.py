# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:51:16 2024

@author: Administrator
"""

from sklearn.ensemble import StackingRegressor #导入 Stacking 集成回归模块
from sklearn.linear_model import LinearRegression #导入 Logistic 线性回归模块
from sklearn.ensemble import RandomForestRegressor #导入随机森林回归模块
from sklearn.ensemble import AdaBoostRegressor #导入 AdaBoost 回归模块
from sklearn.ensemble import GradientBoostingRegressor #导入梯度提升树模块
#导入绘图库
import matplotlib.pyplot as plt 
from sklearn.metrics import PredictionErrorDisplay 
import numpy as np #导入科学计算库
import time #导入时间模块
#导入交叉验证模块
from sklearn.model_selection import cross_validate, cross_val_predict 
# 构造数据
a,b,c=3,-5,8 #设置参数真值
#真值
x = np.linspace(-5,5,100) 
y_real = a * np.power(x,2) + b * x + c 
#生成仿真数据
y = y_real + np.random.normal(0,5,100) 
x = x.reshape(-1,1) 
y_real = y_real.reshape(-1,1) 
y = y.reshape(-1,1) 
#显示结果
LR = LinearRegression() #线性回归
RFR = RandomForestRegressor() #随机森林
ABR = AdaBoostRegressor() #AdaBoost 
GBR = GradientBoostingRegressor() #梯度提升树
estimators = [ 
 ('Random Forest', RFR), 
 ('AdaBoost', ABR), 
 ('Gradient Boosting', GBR)] 
SR = StackingRegressor(estimators=estimators, final_estimator=LR) 
# 评估并输出结果
fig, axs = plt.subplots(2, 2, figsize=(10, 8)) 
axs = np.ravel(axs) 
for ax, (name, est) in zip(axs, estimators + [('Stacking Regressor', SR)]): 
    scorers = {'R2': 'r2', 'MAE': 'neg_mean_absolute_error'} 
    start_time = time.time() 
    scores = cross_validate(est, x, y, scoring=list(scorers.values()), n_jobs=-1,verbose=0) 
    elapsed_time = time.time() - start_time 
    y_pred = cross_val_predict(est, x, y, n_jobs=-1, verbose=0) 
    scores = { 
        key: (f"{np.abs(np.mean(scores[f'test_{value}'])):.2f} +- "f"{np.std(scores[f'test_{value}']):.2f}") 
    for key, value in scorers.items() 
 } 
    display = PredictionErrorDisplay.from_predictions( 
        y_true=y, 
        y_pred=y_pred, 
        kind='actual_vs_predicted',
        ax=ax, 
        scatter_kwargs={'alpha': 0.5, 'color': 'tab:green'}, 
        line_kwargs={'color': 'tab:red'}, 
 ) 
    ax.set_title(f'{name}({elapsed_time:.2f} seconds)') 
    for name, score in scores.items(): 
        ax.plot([], [], ' ', label=f'{name}: {score}') 
    ax.legend(loc='upper left') 
plt.tight_layout() 
plt.subplots_adjust(top=0.9) 
plt.show() 
plt.plot(x, y) 
plt.xlabel('Predicted values') 
plt.ylabel('Actual values') 
plt.show()