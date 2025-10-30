# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:46:00 2024

@author: Administrator
"""

from sklearn.ensemble import GradientBoostingRegressor #导入梯度提升模块
import matplotlib.pyplot as plt #导入绘图模块
import numpy as np #导入科学计算模块
from sklearn import datasets #导入数据集
from sklearn.inspection import permutation_importance #导入特征重要性评估模块
from sklearn.metrics import mean_squared_error #导入标准差评价指标模块
from sklearn.model_selection import train_test_split #导入数据划分模块
# 构建数据集
diabetes = datasets.load_diabetes() 
x, y = diabetes.data, diabetes.target 
# 数据处理
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=13) 
# 构建梯度提升树回归模型
params = {'n_estimators': 500,'max_depth': 4,'min_samples_split': 5,'learning_rate': 
0.01,'loss': 'squared_error'} 
GBR = GradientBoostingRegressor(**params) 
# 训练梯度提升树回归模型
GBR.fit(x_train, y_train) 
# 计算与输出梯度提升树回归模型相应的标准差
MSE = mean_squared_error(y_test, GBR.predict(x_test)) 
print('Mean squared error (MSE): {:.4f}'.format(MSE)) 
# 训练偏差（staged_predict()函数用于返回每个训练轮次的预测结果）
test_score = np.zeros((params['n_estimators'],), dtype=np.float64) 
for i, y_pred in enumerate(GBR.staged_predict(x_test)): 
    test_score[i] = mean_squared_error(y_test, y_pred) 
plt.figure(1) 
plt.plot(np.arange(params['n_estimators']) + 1,GBR.train_score_,'b-',label='Training Deviance') 
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-', label='Test Deviance') 
plt.legend(loc='best') 
plt.xlabel('Iterations') 
plt.ylabel('Deviance') 
plt.grid(True) 
plt.show() 
# 特征重要性
# 特征重要性（不纯度）
feature_importance = GBR.feature_importances_ 
sorted_idx = np.argsort(feature_importance) 
pos = np.arange(sorted_idx.shape[0]) + 0.5 
plt.figure(2) 
plt.barh(pos, feature_importance[sorted_idx], align='center') 
plt.yticks(pos, np.array(diabetes.feature_names)[sorted_idx]) 
# 特征重要性（排列）
plt.figure(3) 
PI = permutation_importance(GBR, x_test, y_test, n_repeats=10, random_state=42, n_jobs=2) 
sorted_idx = PI.importances_mean.argsort() 
plt.boxplot(PI.importances[sorted_idx].T,vert=False,labels=np.array(diabetes.feature_names)[sorted_idx]) 
plt.show()