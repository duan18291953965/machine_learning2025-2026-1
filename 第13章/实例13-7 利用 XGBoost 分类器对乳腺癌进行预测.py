# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:47:22 2024

@author: Administrator
"""

import numpy as np 
from sklearn.datasets import load_breast_cancer #导入数据集
from sklearn.model_selection import train_test_split #导入数据划分模块
import xgboost as xgb #导入 XGBoost 模块
from xgboost.sklearn import XGBClassifier #导入 XGBoost 分类模块
#导入绘图库
import matplotlib.pyplot as plt 
#导入评价指标模块
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import roc_curve 
from xgboost import plot_importance #导入特征重要性绘制模块
import time #导入时间模块
#加载数据
Cancer=load_breast_cancer() 
x = Cancer.data #特征值
y = Cancer.target #目标值
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=1) 
Acc_XGB_1 = [] 
Acc_XGB_2 = [] 
Time_XGB = [] 
# 构建 XGBoost 分类器
params={'n_estimators':300, 
 'num_class':2, 
 'booster':'gbtree', 
 'objective': 'multi:softmax', 
 'max_depth':5, 
 'colsample_bytree':0.75, 
 'min_child_weight':1, 
 'max_delta_step':0, 
 'seed':0, 
 'gamma':0.15, 
 'learning_rate' : 0.01} 
XGB = XGBClassifier(**params) 
start = time.time() 
XGB.fit(x_train, y_train,eval_set=[(x_train,y_train),(x_test,y_test)]) 
Time_XGB.append(time.time() - start) 
#测试 XGBoost 分类器
#AUC（训练数据）
Acc_XGB_1.append(roc_auc_score(y_train, XGB.predict(x_train))) 
#AUC（测试数据）
Acc_XGB_1.append(roc_auc_score(y_test, XGB.predict(x_test))) 
#预测精度
Acc_XGB_1.append(XGB.score(x_test,y_test)) 
# 利用交叉验证方式确定最优参数
dtrain = xgb.DMatrix(x,label=y) 
xgb_param = XGB.get_xgb_params() 
XGB_CV = xgb.cv(xgb_param, dtrain, num_boost_round=5000, nfold=3, metrics=['auc'], early_stopping_rounds=10, stratified=True) 
#显示交叉验证评价指标
print('交叉验证评价指标:') 
print(XGB_CV) 
#更新基学习器数并重新训练 XGBoost 分类器
XGB.set_params(n_estimators=XGB_CV.shape[0]) 
start = time.time() 
XGB.fit(x_train, y_train,eval_set=[(x_train,y_train),(x_test,y_test)]) 
Time_XGB.append(time.time() - start) 
#测试 XGBoost 分类器
#AUC（训练数据）
Acc_XGB_2.append(roc_auc_score(y_train, XGB.predict(x_train))) 
#AUC（测试数据）
Acc_XGB_2.append(roc_auc_score(y_test, XGB.predict(x_test))) 
#预测精度
Acc_XGB_2.append(XGB.score(x_test,y_test)) 
#显示结果
#运行时间对比
bar_width = 0.4 
plt.figure(1) 
plt.bar(range(2), Time_XGB, bar_width, color=['r','c'], label=['n_estimators=300','n_estimators=' + str(XGB_CV.shape[0])]) 
plt.legend(loc="best") 
plt.grid(True) 
plt.xlabel("Estimators") 
plt.ylabel("Time") 
plt.show() 
#精度对比
plt.figure(2) 
index = np.arange(3) 
plt.bar(index, Acc_XGB_1, bar_width, color='r') 
plt.bar(index + bar_width, Acc_XGB_2, bar_width, color='c',tick_label = 
['Train(AUC)','Test(AUC)','Test(Accuracy)']) 
plt.legend(labels = ['n_estimators=100','n_estimators='+str(XGB_CV.shape[0])], 
loc='lower left') 
plt.grid(True) 
plt.xlabel("Metric") 
plt.ylabel("Value") 
plt.show() 
# 绘制 AUC 曲线
fpr, tpr, T = roc_curve(y_test,XGB.predict(x_test)) 
auc_score = roc_auc_score(y_test, XGB.predict(x_test)) 
plt.plot(fpr,tpr,label= f'AUC = {auc_score:.4f}') 
plt.plot([0,1],[0,1],linestyle='--',color='r',label = 'Random Classifier') 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate') 
plt.legend() 
plt.grid(True) 
plt.show() 
# 显示特征重要性
plot_importance(XGB,height=0.8)