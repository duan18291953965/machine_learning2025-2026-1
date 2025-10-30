# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 17:03:46 2024

@author: Administrator
"""

#导入绘图库与鸢尾花数据集
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
#导入Logistic回归库
from sklearn.linear_model import LogisticRegression 
#加载数据集并调整类别标记
Iris=load_iris()
x=Iris.data
y=Iris.target
ix = [i for i in range(len(y)) if y[i]!=2]
x_new = x[ix,:]
y_new = y[ix]
#输出数据基本信息
print('数据基本信息: {0}; Class_1: {1}; Class_2: {2}'.format(x_new.shape, y_new[y_new==1].sum(), y_new[y_new==0].shape[0]))
print('特征名称:',Iris.feature_names)
#将数据集划分为训练数据与测试数据
X_train,X_test,Y_train,Y_test=train_test_split(x_new,y_new,test_size=0.3,random_state=0)
#构建Logistic回归模型
LR = LogisticRegression()
#模型训练
LR.fit(X_train,Y_train)
#模型预测
pro = LR.predict_proba(X_test) #利用测试数据预测概率
acc = LR.score(X_test,Y_test) #利用测试数据求取精度
#显示前10行预测概率
print('前10个样本的预测概率:',pro[:10])
#显示前10个样本类别标记
print('前10个样本的预测概率:',Y_test[:10])
#输出模型准确率
print('模型在测试集上的预测精度:',acc)
#L1正则化
LR_L1=LogisticRegression(penalty='l1',solver='liblinear',C=0.02,max_iter=1000) 
LR_L1.fit(x_new,y_new)
print('L1正则化系数:',LR_L1.coef_)
print('非零L1正则化系数:',(LR_L1.coef_!=0).sum(axis=1))
#L2正则化
LR_L2=LogisticRegression(penalty='l2',solver='liblinear',C=0.02,max_iter=1000)
LR_L2.fit(x_new,y_new)
print('L2正则化系数:',LR_L2.coef_)
print('非零L2正则化系数:',(LR_L2.coef_!=0).sum(axis=1))
#比较L1与L2正则化相应的精度
Acc = []
LR_L1=LogisticRegression(penalty='l1',solver='liblinear',C=0.02,max_iter=1000)
LR_L1.fit(X_train,Y_train)
Acc.append(LR_L1.score(X_test,Y_test)) #L1正则化相应的精度
LR_L2=LogisticRegression(penalty='l2',solver='liblinear',C=0.02,max_iter=1000)
LR_L2.fit(X_train,Y_train)
Acc.append(LR_L2.score(X_test,Y_test)) #L1正则化相应的精度
#画出L1与L2正则化相应的精度对比柱状图
plt.bar(range(len(Acc)),Acc,color=['red','lightgreen'],tick_label=['L1','L2'])
for i,k in zip(range(len(Acc)),Acc):
    plt.text(i,k,str(k))
plt.xlabel('Regularization')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()


