# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:34:38 2024

@author: Administrator
"""

import numpy as np #导入科学计算库
from sklearn.preprocessing import MinMaxScaler #导入归一化模块
from sklearn.naive_bayes import MultinomialNB #导入多项式朴素贝叶斯库
from sklearn.model_selection import train_test_split #导入数据划分库
from sklearn.datasets import make_blobs #导入数据库
#构造数据
X, y = make_blobs(n_samples= 1000, centers=[[0.0, 0.1], [2.0, 3.0]], cluster_std=[0.4,0.5], random_state=0) 
#数据归一化
mms = MinMaxScaler().fit(X)
X = mms.transform(X)
#将数据划分为训练数据与测试数据
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.3,random_state=100)
#构建多项式朴素贝叶斯分类器
mnb = MultinomialNB().fit(Xtrain, Ytrain) 
print('类先验概率:', np.exp(mnb.class_log_prior_)) #查看先验概率的对数
print('类条件概率:',np.exp(mnb.feature_log_prob_)) #查看类条件概率
#利用测试数据测试模型的精度
print('预测精度:', mnb.score(Xtest,Ytest))


from sklearn.preprocessing import KBinsDiscretizer #数据离散化模块
#数据离散化
kbd = KBinsDiscretizer(n_bins=10, encode='onehot').fit(X) 
Xtrain_new = kbd.transform(Xtrain) 
Xtest_new = kbd.transform(Xtest) 
#查看离散化化数据基本结构
print('数据基本结构:', Xtrain_new.shape)
#构建多项式朴素贝叶斯分类器并利用离散化的训练数据进行训练
mnb = MultinomialNB().fit(Xtrain_new, Ytrain) 
#利用离散化测试数据测试模型的精度
print('预测精度:',mnb.score(Xtest_new,Ytest))
