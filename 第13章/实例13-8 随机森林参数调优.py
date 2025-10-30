# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:48:46 2024

@author: Administrator
"""

from sklearn.ensemble import RandomForestClassifier #导入随机森林模块
from sklearn.model_selection import train_test_split #导入数据划分模块
from sklearn.tree import DecisionTreeClassifier #导入决策树模块
from sklearn.datasets import load_wine #导入红酒数据集
from sklearn.model_selection import GridSearchCV #导入网格式参数优化模块
#导入绘图模块
import matplotlib.pyplot as plt 
import matplotlib as mpl 
import numpy as np #导入科学计算库
#加载数据
wine = load_wine() 
x,y = wine.data,wine.target 
#将数据划分为训练数据与测试数据
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1) 
# 决策树参数优化
# 设置参数搜索范围
param_grid = {'criterion': ['entropy', 'gini'], 
 'max_depth': [2, 3, 4, 5, 6, 7, 8], 
 'min_samples_split': [4, 8, 12, 16, 18, 20]} 
# 构建决策树模型
DTC = DecisionTreeClassifier() 
# 参数优化
DTC_CV = GridSearchCV(estimator=DTC, param_grid=param_grid, scoring='accuracy', cv=3) 
# 训练决策树模型
DTC_CV.fit(x_train, y_train) 
# 显示最优参数信息
print('决策树最优参数：',DTC_CV.best_params_) 
# 显示最优模型
print('决策树最优模型：',DTC_CV.best_estimator_) 
# 利用最优参数构建决策树模型 
DTC.set_params(**DTC_CV.best_params_) 
DTC.fit(x_train,y_train) 
# 评估决策树模型
print('决策树预测精度:',DTC.score(x_test,y_test)) 
# 随机森林参数优化
# 设置参数搜索范围
param_grid = { 
 'criterion':['entropy','gini'], 
 'max_depth':[5, 6, 7, 8, 10], 
 'n_estimators':[9, 11, 13, 15], 
 'max_features':[0.3, 0.4, 0.5, 0.7], 
 'min_samples_split':[4, 8, 10, 12, 16] 
} 
# 构建随机森林模型
RFC = RandomForestClassifier() 
# 参数优化
RFC_CV = GridSearchCV(estimator=RFC, param_grid=param_grid,scoring='accuracy', cv=3) 
# 训练随机森林模型
RFC_CV.fit(x_train, y_train) 
# 显示最优参数信息
print('随机森林最优参数：',RFC_CV.best_params_) 
# 显示最优模型
print('随机森林最优模型：',RFC_CV.best_estimator_) 
# 利用最优参数构建随机森林模型 
RFC.set_params(**RFC_CV.best_params_) 
RFC.fit(x_train,y_train) 
# 评估随机森林模型
print('随机森林精度:',RFC.score(x_test,y_test)) 
# 求取特征重要性
print('特征重要性排序如下。') 
importances = RFC.feature_importances_ 
indices = np.argsort(importances)[::-1] 
for f in range(x_train.shape[1]): 
    print('%2d) %-*s %f' % (f + 1, 30, wine.feature_names[indices[f]], importances[indices[f]])) 
# 特征重要性可视化
plt.figure() 
names = [wine.feature_names[i] for i in indices] #特征名称
plt.bar(range(x_train.shape[1]), importances[indices]) 
plt.xticks(range(x_train.shape[1]), names, rotation=20) 
plt.ylabel('Importance') 
plt.xticks(fontsize=8) 
plt.yticks(fontsize=8) 
plt.grid(True) 
plt.show() 
# 选择特征
threshold = 0.13 #特征阈值
x_selected_train = x_train[:, importances > threshold] 
x_selected_test = x_test[:, importances > threshold] 
# 训练随机森林模型
RFC_CV.fit(x_selected_train, y_train) 
# 显示最优参数信息
print('随机森林最优参数：',RFC_CV.best_params_) 
# 显示最优模型
print('随机森林最优模型：',RFC_CV.best_estimator_) 
# 利用最优参数构建随机森林模型 
RFC.set_params(**RFC_CV.best_params_) 
RFC.fit(x_selected_train,y_train) 
# 评估随机森林模型
print('随机森林精度:',RFC.score(x_selected_test,y_test)) 
#绘制分类效果图
x_Min, x_Max = x_selected_test[:,0].min(), x_selected_test[:,0].max() 
y_Min, y_Max = x_selected_test[:,1].min(), x_selected_test[:,1].max() 
xx,yy=np.meshgrid(np.linspace(x_Min, x_Max, 50),np.linspace(y_Min, y_Max, 50)) #生成网格点
GridTest=np.stack((xx.flat, yy.flat), axis=1) #测试点
y_pred=RFC.predict(GridTest) #预测测试点所属类别
CmBg = mpl.colors.ListedColormap(['r', 'y','c']) #背景颜色
plt.figure() 
plt.xlim(x_Min, x_Max);plt.ylim(y_Min, y_Max) #设置坐标范围
plt.pcolormesh(xx, yy, y_pred.reshape(xx.shape), cmap=CmBg) #绘制网格背景
plt.scatter(x_selected_test[y_test==0, 0], x_selected_test[y_test==0, 1], c='k', 
marker='o', linewidths=1, edgecolors='w', label='Class1') 
plt.scatter(x_selected_test[y_test==1, 0], x_selected_test[y_test==1, 1], c='k', 
marker='s', linewidths=1, edgecolors='w', label='Class2') 
plt.scatter(x_selected_test[y_test==2, 0], x_selected_test[y_test==2, 1], c='k', 
marker='^', linewidths=1, edgecolors='w', label='Class3') 
plt.xlabel('X1') 
plt.ylabel('X2') 
plt.legend(loc='best') 
plt.grid(True) 
plt.show()