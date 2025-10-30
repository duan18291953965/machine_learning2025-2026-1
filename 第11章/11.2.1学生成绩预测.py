# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 12:56:42 2024

@author: Administrator
"""

#导入科学计算相关库
import pandas as pd 
import numpy as np 
#导入绘图库
from matplotlib import pyplot as plt 
import seaborn as sns 
from sklearn.neural_network import MLPClassifier 
from sklearn.model_selection import train_test_split #导入数据划分模块
from sklearn.preprocessing import StandardScaler #导入数据标准化模块
from sklearn.model_selection import GridSearchCV #导入网格式参数调优模块
from sklearn.inspection import permutation_importance #导入特征重要性评估模块
#加载数据
data = pd.read_csv( './/data//Expanded_data_with_more_features.csv',encoding=u'gbk') 
#数据编码
edu_encoding = { 
 'TestPrep' : { 
 'completed' : 1, 
 'none' : 0 
 }, 
 'ParentMaritalStatus' : { 
 'widowed' : 3, 
 'divorced' : 2, 
 'married' : 1, 
 'single' : 0 
 }, 
 'PracticeSport' : { 
 'regularly' : 2, 
 'sometimes' : 1, 
 'never' : 0 
 }, 
 'IsFirstChild' : { 
 'yes' : 1, 
 'no' : 0 
 }, 
} 
for column in data: 
    if column in edu_encoding.keys(): 
        try: 
            data[column] = data[column].apply( lambda x : edu_encoding[column][x] ) 
        except: 
            print(f"Skipped {column}") 
print('数据基本信息:',data.shape) #显示数据基本信息（样本数与特征数）
x = data.drop('Score' , axis= 1) 
y = data['Score'] 
#将成绩转换为好或差
y[y<60] = 0 
y[y>=60] = 1 
#特征相关性分析
plt.figure() 
plt.rcParams['figure.figsize'] = (20, 15) 
plt.style.use('ggplot') 
sns.heatmap(x.corr(), annot = True, cmap = 'Wistia') 
plt.show() 
#数据标准化
scaler = StandardScaler() 
x_ = scaler.fit_transform(x) 
#将数据划分为训练数据与测试数据
x_train, x_test, y_train, y_test = train_test_split(x_,y,test_size=0.4) 
# 求取最优参数
param_grid = { 
 'hidden_layer_sizes': [(5,), (10,), (5, 5)], 
 'activation': ['relu', 'tanh'], 
 'solver':['lbfgs','sgd','adam'], 
 'alpha': [0.0001, 0.001, 0.01], 
 'learning_rate': ['constant', 'invscaling', 'adaptive'] 
} 
# 使用 MLPClassifier 构建 MLP（Multilayer Perceptron，多层感知器）模型
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
print("预测精度:", NN.score(x_test,y_test)) 
# 获取特征重要性
feature_importance = abs(NN.coefs_[0]) 
# 特征的重要性
PI = permutation_importance(NN, x_test, y_test, n_repeats=10, random_state=1, n_jobs=2) 
sorted_idx = PI.importances_mean.argsort() 
plt.boxplot(PI.importances[sorted_idx].T,vert=False,labels=np.array(data.columns) 
[sorted_idx]) 
plt.show()