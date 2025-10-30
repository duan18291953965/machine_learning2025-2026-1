# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:35:54 2024

@author: Administrator
"""

import matplotlib.pyplot as plt #导入绘图库
from itertools import product #导入迭代器处理模块
from sklearn import datasets #导入数据集
from sklearn.tree import DecisionTreeClassifier #导入决策树模块
from sklearn.neighbors import KNeighborsClassifier #导入 K 近邻分类模块
from sklearn.svm import SVC #导入 SVC 模块
from sklearn.ensemble import VotingClassifier #导入分类学习器模块
from sklearn.inspection import DecisionBoundaryDisplay #导入决策边界显示模块
from sklearn.model_selection import train_test_split #导入数据划分模块
# 加载数据集
iris = datasets.load_iris() 
x = iris.data[:, [0, 2]] 
y = iris.target 
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=13) 
# 训练分类器
dt_model = DecisionTreeClassifier(max_depth=4).fit(x_train,y_train) #决策树
kn_model = KNeighborsClassifier(n_neighbors=7).fit(x_train,y_train) #K 近邻
#支持向量机
svm_model = SVC(gamma=0.1, kernel="rbf", probability=True).fit(x_train,y_train) 
#集成学习（加权投票）
vc_model = VotingClassifier( 
 estimators=[("dt", dt_model), ("knn", kn_model), ("svc", svm_model)], 
 voting="soft", 
 weights=[0.2, 0.1, 0.7], 
) 
vc_model.fit(x_train,y_train) 
# 绘制决策边界
dt_score = format(dt_model.score(x_test,y_test),'.2f') 
kn_score = format(kn_model.score(x_test,y_test),'.2f') 
svm_score = format(svm_model.score(x_test,y_test),'.2f') 
vc_score = format(vc_model.score(x_test,y_test),'.2f') 
f, axarr = plt.subplots(2, 2,sharex="col", sharey=False, figsize=(10, 8)) 
for ix, model, name in zip( 
 product([0, 1], [0, 1]), 
 [dt_model, kn_model, svm_model, vc_model], 
 ['Decision_Tree (depth=4)' + ' Accuracy:' + str(dt_score), 
 'KNN(k=7)' + ' Accuracy:' + str(kn_score), 
 'SVM(kernel="rbf")' + ' Accuracy:' + str(svm_score), 
 'Voting(Weight=(0.2,0.1,0.7))' + ' Accuracy:' + str(vc_score)], 
): 
    DecisionBoundaryDisplay.from_estimator( 
        model, x, alpha=0.4, ax=axarr[ix[0], ix[1]], response_method="predict") 
    axarr[ix[0], ix[1]].scatter(x[:, 0], x[:, 1], c=y, s=20, edgecolor="k") 
    axarr[ix[0], ix[1]].set_title(name) 
plt.show()