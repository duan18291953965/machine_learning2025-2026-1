# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:53:11 2024

@author: Administrator
"""
import numpy as np #导入科学计算库
from sklearn.naive_bayes import BernoulliNB #导入伯努利朴素贝叶斯库
from sklearn.model_selection import train_test_split #导入数据划分库
import  matplotlib.pyplot as plt #导入绘图库
from sklearn.datasets import load_digits #导入手写数字数据库
from sklearn import preprocessing #导入数据预处理库
import matplotlib.pyplot as plt #导入绘图库
#加载数据
digits=load_digits()
x=digits.data
y=digits.target
print(x)
print(y)
# 归一化处理
transfer = preprocessing.MinMaxScaler()
x=transfer.fit_transform(x)
# 数据维度 属性1：1000   属性2 : 1.8  1.9 ---->数据统一 权重、

#显示手写数字样本示例
plt.figure()
for i in range(16):
     plt.subplot(4,4,i+1)
     plt.imshow(digits.images[i])
plt.show()
# #显示数据维度
print(x.shape)  #(1797, 64)
print(y.shape)  #(1797,)
# #进行训练集和测试集切分
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25)
#构建伯努利朴素贝叶斯分类器
BNB=BernoulliNB()
#训练伯努利朴素贝叶斯分类器
BNB.fit(x_train,y_train)
# #测试伯努利朴素贝叶斯分类器
print('测试精度: %.2f' % BNB.score(x_test, y_test))   #x_test--->x_test的预测值   和真实的y_test
#测试伯努利朴素贝叶斯在不同特征二值化时的精度
min_x=min(np.min(x_train.ravel()),np.min(x_test.ravel()))
max_x=max(np.max(x_train.ravel()),np.max(x_test.ravel()))
bin_list=np.linspace(min_x,max_x,endpoint=True,num=50)
train_accuracy=[]
test_accuracy=[]
for b in bin_list:
    BNB=BernoulliNB(binarize=b)
    BNB.fit(x_train,y_train)
    train_accuracy.append(BNB.score(x_train,y_train))
    test_accuracy.append(BNB.score(x_test, y_test))
#显示结果
plt.figure()
plt.plot(bin_list,train_accuracy,label="Training_Accuracy")
plt.plot(bin_list,test_accuracy,label="Testing_Accuracy")
plt.xlabel('Binarization')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend(loc="best")
plt.show()

