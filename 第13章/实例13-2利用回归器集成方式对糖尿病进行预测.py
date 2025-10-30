# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:37:59 2024

@author: Administrator
"""

import matplotlib.pyplot as plt #导入绘图库
from sklearn.ensemble import GradientBoostingRegressor #导入梯度提升树模块
from sklearn.ensemble import RandomForestRegressor #导入随机森林模块
from sklearn.linear_model import LinearRegression #导入线性回归模块
from sklearn.ensemble import VotingRegressor #导入集成学习模块
from sklearn.datasets import load_diabetes #导入糖尿病数据集
from sklearn.model_selection import train_test_split #导入数据划分模块
from sklearn.metrics import r2_score #导入 R2 分数模块
#加载糖尿病数据
diabetes = load_diabetes() 
x = diabetes.data 
y = diabetes.target 
#将数据划分为训练数据与测试数据
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3) 
#构建个体学习器
GB = GradientBoostingRegressor() 
RF = RandomForestRegressor() 
LR = LinearRegression() 
#训练个体学习器
GB.fit(x_train,y_train) 
RF.fit(x_train,y_train) 
LR.fit(x_train,y_train) 
#利用投票方式对个体学习器进行集成
EL = VotingRegressor([("gb", GB), ("rf", RF), ("lr", LR)]) 
#训练集成学习器
EL.fit(x_train,y_train) 
#测试个体学习器与集成学习器
xt = x_test[:20] #选择 20 个样本
GB_Pred = GB.predict(xt) 
RF_Pred= RF.predict(xt) 
LR_Pred = LR.predict(xt) 
EL_Pred = EL.predict(xt) 
#显示评价指标值
ACC = [] 
ACC.append(r2_score(y_test[:20], GB_Pred)) 
ACC.append(r2_score(y_test[:20], RF_Pred)) 
ACC.append(r2_score(y_test[:20], LR_Pred)) 
ACC.append(r2_score(y_test[:20], EL_Pred)) 
#显示对比结果
plt.figure(1) 
label = ['GradientBoostingRegressor','RandomForestRegressor','LinearRegression', 
'VotingRegressor'] 
plt.bar(range(4),height=ACC,color = ['r','g','b','c'],tick_label=label,width = 0.4) 
plt.xlabel('Method') 
plt.ylabel('Accuracy') 
for xx, yy in zip(range(4),ACC): 
    plt.text(xx, yy, format(yy,'.2f'), ha='center', fontsize=10) 
plt.grid(True) 
plt.show() 
#可视化结果
plt.figure(2) 
plt.plot(y_test[:20], 'ko', label='Ground_Truth') 
plt.plot(GB_Pred, 'gd', label='GradientBoostingRegressor') 
plt.plot(RF_Pred, 'b^', label='RandomForestRegressor') 
plt.plot(LR_Pred, 'ys', label='LinearRegression') 
plt.plot(EL_Pred, 'r*', ms=10, label='VotingRegressor') 
plt.tick_params(axis="x", which='both', bottom=False, top=False, labelbottom=False) 
plt.ylabel('Predicted_Values') 
plt.xlabel('Testing samples') 
plt.legend(loc='best') 
plt.grid(True) 
plt.show()