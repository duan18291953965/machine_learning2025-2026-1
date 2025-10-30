# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:50:14 2024

@author: Administrator
"""

from sklearn.linear_model import LogisticRegression #导入 Logistic 回归模块
from sklearn.neighbors import KNeighborsClassifier #导入 K 近邻模块
from sklearn.naive_bayes import GaussianNB #导入高斯朴素贝叶斯模块
from sklearn.ensemble import RandomForestClassifier #导入随机森林分类模块
from sklearn.ensemble import StackingClassifier #导入 Stacking 分类框架模块
from sklearn import model_selection #导入模型选择模块
from sklearn.datasets import load_iris #导入鸢尾花数据集
from sklearn.preprocessing import StandardScaler #导入数据标准化模块
#加载数据
iris = load_iris() 
X, y = iris.data, iris.target 
#标准化
scaler = StandardScaler() 
X = scaler.fit_transform(X) 
#初始分类器参数确定
KN=KNeighborsClassifier() #K 近邻
RF=RandomForestClassifier() #随机森林
GN=GaussianNB() #高斯朴素贝叶斯
LR=LogisticRegression() #Logistic 回归
#利用 Stacking 集成框架构建集成学习器
sclf=StackingClassifier(estimators=[KN,RF,GN],final_estimator=LR)
#精度对比
for clf, label in zip([KN,RF,GN,LR],['KNeighborsClassifier', 'RandomForestClassifier', 'GaussianNBClassifier', 'StackingClassifier']): 
    scores = model_selection.cross_val_score(clf, X, y, cv=3, scoring='accuracy') 
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))