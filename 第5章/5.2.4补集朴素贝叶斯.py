# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:56:46 2024

@author: Administrator
"""

from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB #导入高斯、多项式与伯努利朴素贝叶斯库
from sklearn.naive_bayes import ComplementNB #导入补集朴素贝叶斯库
from sklearn.preprocessing import KBinsDiscretizer #导入特征离散化库
from sklearn.model_selection import train_test_split #导入数据划分库
from sklearn.metrics import recall_score, roc_auc_score #导入评价指标库
from sklearn.datasets import make_blobs #导入make_blobs数据库
#构造数据不均衡样本数据集
x, y = make_blobs(n_samples= [100000,500], centers=[[0.0, 0.1], [3.0, 5.0]], cluster_std=[1.8,1.5], random_state=0) 
#设置不同类型的朴素贝叶斯模型
nb_names=["Gaussian","Multinomial","Bernoulli","Complement"]
nb_models=[GaussianNB(),MultinomialNB(),BernoulliNB(),ComplementNB()]
#求取四种朴素贝叶斯模型相应的精度、召回率与AUC值
for nb,name in zip(nb_models,nb_names):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=10)
    #离散化特征值以用于多项式、伯努利与补集朴素贝叶斯
    if name!="Gaussian":
        kbs = KBinsDiscretizer(n_bins=10,encode="onehot").fit(x)
        x_train = kbs.transform(x_train)
        x_test = kbs.transform(x_test)
    nb.fit(x_train, y_train) 
#输出结果
    print(name)
    print("\tAccuracy:{:.3f}".format(nb.score(x_test,y_test)))
    print("\tRecall:{:.3f}".format(recall_score(y_test,nb.predict(x_test))))
    print("\tAUC:{:.3f}".format(roc_auc_score(y_test,nb.predict_proba(x_test)[:,1]))) 
