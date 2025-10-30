# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:01:20 2024

@author: Administrator
"""

#导入数据分析库
import pandas as pd 
import numpy as np 
#导入绘图库
from matplotlib import pyplot as plt 
import seaborn as sns 
import matplotlib as mpl 
from sklearn.preprocessing import StandardScaler #导入数据标准化模块
from sklearn.neural_network import MLPClassifier #导入多层感知机模块
from sklearn.model_selection import train_test_split #导入数据划分模块
from sklearn.decomposition import PCA #导入主成分分析模块
from sklearn.model_selection import GridSearchCV #导入网格式参数调优模块
#加载数据
data = pd.read_csv( './/data//dataset_heart.csv',encoding=u'gbk') 
#分离特征与类别标记
x = data.drop('target', axis= 1) 
y = data['target']-1 
# 特征相关性分析
corr = x[x.columns[0:]].corr() 
plt.figure(figsize=(10,8)) 
ax = sns.heatmap( 
 corr, 
 vmin=-1, vmax=1, center=0, 
 cmap=sns.diverging_palette(20, 220, n=200), 
 square=False, annot=True,fmt='.1f') 
ax.set_xticklabels( 
 ax.get_xticklabels(), 
 rotation=30, 
 horizontalalignment='right' 
) 
#数据标准化
scaler = StandardScaler() 
x_ = scaler.fit_transform(x) 
#将数据划分为训练数据与测试数据
x_train, x_test, y_train, y_test = train_test_split(x_,y,test_size=0.3,random_state=5) 
# 求取最优参数
param_grid = { 
 'hidden_layer_sizes': [(5,), (10,), (5, 5)], 
 'activation': ['relu', 'tanh'], 
 'alpha': [0.0001, 0.001, 0.01], 
 'learning_rate': ['constant', 'invscaling', 'adaptive'] 
} 
# 构建 MLP 模型
NN = MLPClassifier() 
# 采用 GridSearchCV 调参
grid_search = GridSearchCV(NN, param_grid, cv=5) 
grid_search.fit(x_train, y_train) 
# 输出最优参数与最高精度
print("最优参数:", grid_search.best_params_) 
print("最高精度:", grid_search.best_score_) 
# 采用最优参数进行模型训练与测试
NN.set_params(**grid_search.best_params_) 
NN.fit(x_train,y_train) 
# 输出测试精度
print("预测精度", NN.score(x_test,y_test)) 
# 主成分分析
pca =PCA(n_components=2, whiten=True).fit(x_train) 
x_train_pca = pca.transform(x_train) 
x_test_pca = pca.transform(x_test) 
# 构建 MLP 模型
NN = MLPClassifier() 
# 采用 GridSearchCV 调参
grid_search = GridSearchCV(NN, param_grid, cv=5) 
grid_search.fit(x_train_pca, y_train) 
# 输出最优参数与最高精度
print("最优参数(PCA):", grid_search.best_params_) 
print("最高精度(PCA):", grid_search.best_score_) 
# 采用最优参数进行模型训练与测试
NN.set_params(**grid_search.best_params_) 
NN.fit(x_train_pca,y_train) 
# 输出测试精度
print("预测精度(PCA):", NN.score(x_test_pca,y_test)) 
#显示分类效果
plt.figure() 
x_min,x_max = x_test_pca[:,0].min(),x_test_pca[:,0].max() 
y_min,y_max = x_test_pca[:,1].min(),x_test_pca[:,1].max() 
xx,yy = np.meshgrid(np.linspace(x_min, x_max, 200),np.linspace(y_min, y_max, 200)) 
grid_test = np.stack((xx.flat, yy.flat), axis=1) 
y_pred = NN.predict(grid_test) 
cm_pt = mpl.colors.ListedColormap(['w', 'g']) 
cm_bg = mpl.colors.ListedColormap(['r', 'y']) 
plt.xlim(x_min, x_max) 
plt.ylim(y_min, y_max) 
plt.pcolormesh(xx, yy, y_pred.reshape(xx.shape), cmap=cm_bg) 
plt.scatter(x_test_pca[:,0],x_test_pca[:,1],c=y_test,cmap=cm_pt,marker='o', 
linewidths=1,edgecolors='k') 
plt.grid(True) 
plt.show() 
