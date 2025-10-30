# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:38:38 2024

@author: Administrator
"""

from sklearn.datasets import fetch_20newsgroups #导入数据
from sklearn.model_selection import train_test_split #导入数据划分库
from sklearn.feature_extraction.text import TfidfVectorizer #导入词频统计与向量化库
from sklearn.naive_bayes import MultinomialNB #导入多项式朴素贝叶斯库
news = fetch_20newsgroups(subset='all') #下载数据（包括训练数据与测试数据）
x = news.data
y = news.target
#将数据划分为训练数据与测试数据
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.4)
#词频统计与向量化
tf = TfidfVectorizer()
x_train = tf.fit_transform(x_train)
x_test = tf.transform(x_test)
#print(tf.get_feature_names())
#构建多项式朴素贝叶斯分类器
MNB= MultinomialNB(alpha=1.0)
#训练多项式朴素贝叶斯分类器
MNB.fit(x_train,y_train)
#测试多项式朴素贝叶斯分类器
#输出测试精度
print('预测精度:', MNB.score(x_test,y_test))

