# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:41:56 2024

@author: Administrator
"""

from sklearn.model_selection import train_test_split #导入数据划分模块
import matplotlib.pyplot as plt #导入绘图库
import numpy as np #导入科学计算库
import time #导入时间模块
from sklearn.ensemble import GradientBoostingClassifier #导入梯度提升树分类器模块
from sklearn import datasets #导入数据集
data_list = [datasets.load_iris(return_X_y=True),datasets.load_breast_cancer 
(return_X_y=True),datasets.make_hastie_10_2(n_samples=500, random_state=0)] 
data_names = ["Iris Data", "Breast_cancer Data", "Hastie Data"] 
Num_GB = [] #保存梯度提升树个体学习器数量（非“早期停止”）
Acc_GB = [] #保存梯度提升树精度（非“早期停止”）
Time_GB = [] #保存梯度提升树时间（非“早期停止”）] 
Num_GBES = [] #保存梯度提升树个体学习器数量（“早期停止”）
Acc_GBES = [] #保存梯度提升树精度（“早期停止”）
Time_GBES = [] #保存梯度提升树时间（“早期停止”）
n_estimators = 100 
for x, y in data_list: 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0) 
 #构建梯度提升树（非“早期停止”）
    GB = GradientBoostingClassifier(n_estimators=n_estimators, random_state=0) 
    start = time.time() 
    GB.fit(x_train, y_train) 
    Time_GB.append(time.time() - start) 
    Acc_GB.append(GB.score(x_test, y_test)) 
    Num_GB.append(GB.n_estimators_) 
 #构建梯度提升树（“早期停止”）
    GBES = GradientBoostingClassifier(n_estimators=n_estimators, validation_fraction=0.2,n_iter_no_change=5, tol=0.01, random_state=0) 
#validation_fraction=0.2 是指在训练过程中将训练数据的 20%作为验证数据以用于模型性能的评估；n_iter_no_change=5 是指如果在连续 5 次迭代中模型的性能没
#有提升则停止模型的训练
    start = time.time() 
    GBES.fit(x_train, y_train) 
    Time_GBES.append(time.time() - start) 
    Acc_GBES.append(GBES.score(x_test, y_test)) 
    Num_GBES.append(GBES.n_estimators_) 
#梯度提升树非“早期停止”与“早期停止”两种情况下的精度对比
bar_width = 0.4 
index = np.arange(0,len(data_list)) 
plt.figure(figsize=(8,4)) 
Bar_GB = plt.bar(index, Acc_GB, bar_width, label='Without early stopping', color='blue') 
for i, b in enumerate(Bar_GB): 
    plt.text(b.get_x() + b.get_width() / 2.0, b.get_height(),'n_est=%d' %Num_GB[i],ha= 'center',va='bottom') 
Bar_GBES = plt.bar(index + bar_width, Acc_GBES, bar_width, label='With early stopping', color='cyan') 
for i, b in enumerate(Bar_GBES): 
    plt.text(b.get_x() + b.get_width() / 2.0, b.get_height(), "n_est=%d" %Num_GBES[i], ha="center",va="bottom") 
plt.xticks(index + bar_width, data_names) 
plt.yticks(np.arange(0, 1.2, 0.1)) 
plt.ylim([0, 1.2]) 
plt.legend(loc="best") 
plt.grid(True) 
plt.xlabel("Datasets") 
plt.ylabel("Accuracy") 
plt.show() 
#梯度提升树非“早期停止”与“早期停止”两种情况下的时间对比
plt.figure(figsize=(8,4)) 
Bar_GB = plt.bar(index, Time_GB, bar_width, label='Without early stopping', color='blue') 
for i, b in enumerate(Bar_GB): 
    plt.text(b.get_x() + b.get_width() / 2.0, b.get_height(),  'n_est=%d' %Num_GB[i],ha='center',va='bottom') 
Bar_GBES = plt.bar(index + bar_width, Time_GBES, bar_width, label='With early stopping', color='cyan') 
for i, b in enumerate(Bar_GBES): 
    plt.text(b.get_x() + b.get_width() / 2.0, b.get_height(),'n_est=%d' %Num_GBES[i], ha='center',va="bottom") 
max_time = np.amax(np.maximum(Time_GB, Time_GBES)) 
plt.xticks(index + bar_width, data_names) 
plt.yticks(np.linspace(0, max_time, 10)) 
plt.ylim([0, max_time]) 
plt.legend(loc="best") 
plt.grid(True) 
plt.xlabel("Datasets") 
plt.ylabel("Time") 
plt.show()