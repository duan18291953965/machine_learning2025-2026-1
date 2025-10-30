# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:02:25 2025

@author: Administrator
"""

import numpy as np #导入科学计算库
from sklearn.decomposition import PCA #导入主成分分析库
import matplotlib.pyplot as plt #导入绘图库
#加载糖尿病数据
from sklearn.datasets import load_diabetes 
Data_diabetes = load_diabetes() 
X = Data_diabetes['data'] 
print(X.shape) #查看样本数与特征数
pca_opt = PCA(n_components=10).fit(X) 
#查看可解释性方差贡献率
print(pca_opt.explained_variance_ratio_) 
#累积可解释性方差贡献率并绘制变化曲线
plt.plot([1,2,3,4,5,6,7,8,9,10],np.cumsum(pca_opt.explained_variance_ratio_)) 
plt.xticks([1,2,3,4,5,6,7,8,9,10]) 
plt.xlabel("Number of components") 
plt.ylabel("Cumulative explained variance") 
plt.show() 
plt.grid(True)