# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 16:27:42 2024

@author: Administrator
"""

import numpy as np #导入科学计算库
#导入绘图库
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split #导入数据划分模块
from sklearn.linear_model import LogisticRegression #导入Logistic回归库
from sklearn.datasets import load_iris #导入鸢尾花数据库
from sklearn.decomposition import PCA #导入主成分分析模块
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA #导入线性分析模块
from sklearn.preprocessing import StandardScaler #导入数据标准化模块
# 加载数据
iris = load_iris()
x = iris.data
y = iris.target
# 数据标准化处理
ds = StandardScaler()
x_ = ds.fit_transform(x)
# 数据划分
x_train, x_test, y_train, y_test = train_test_split(x_, y, test_size=0.3, stratify=y, random_state=0)
# 采用全部特征进行分类
LR = LogisticRegression()
LR.fit(x_train, y_train)
print('采用全部特征的精度:',LR.score(x_test, y_test))
# 利用主成分分析提取特征
pca = PCA(n_components=2)
pca.fit(x_)
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)
print('方差占比:', np.max(np.cumsum(pca.explained_variance_ratio_ *100)))
LR_pca = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
LR_pca.fit(x_train_pca, y_train)
print('PCA提取两维特征时的精度:',LR_pca.score(x_test_pca,y_test))
# 利用线性判别分析提取特征
lda = LDA(n_components=2)
x_train_lda = lda.fit_transform(x_train, y_train)
x_test_lda = lda.fit_transform(x_test, y_test)
print('方差占比:', np.max(np.cumsum(lda.explained_variance_ratio_ *100)))
LR_lda = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
LR_lda.fit(x_train_lda, y_train)
print('LDA提取两维特征时的精度:',LR_lda.score(x_test_lda,y_test))
# 定义分类边界绘制函数
def plot_decision_boundary(X, y, model, resolution=0.01):
    markers = ('o', 's', '^')
    colors = ('red', 'blue', 'green')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min, x2_max, resolution))
    Z = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],alpha=0.8, c=[cmap(idx)], edgecolor='black',marker=markers[idx], label=cl)
# 显示主成分析相应的分类结果       
plt.figure(1)
plot_decision_boundary(x_train_pca, y_train, model=LR_pca)
plt.xlabel('x1_pca')
plt.ylabel('x2_pca')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
# 显示线性判别分析相应的分类结果 
plt.figure(2)
plot_decision_boundary(x_train_lda, y_train, model=LR_lda)
plt.xlabel('x1_lda')
plt.ylabel('x2-lda')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
