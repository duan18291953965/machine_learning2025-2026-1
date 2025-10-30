# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 18:59:03 2025

@author: Administrator
"""

from sklearn.datasets import load_iris 
# 加载鸢尾花数据集
iris = load_iris() 
print("鸢尾花的特征值:\n", iris["data"]) 
print("鸢尾花的目标值：\n", iris.target) 
print("鸢尾花特征名字：\n", iris.feature_names) 
print("鸢尾花类别名字：\n", iris.target_names) 
print("鸢尾花的描述：\n", iris.DESCR)


from sklearn.datasets import make_blobs 
data, label = make_blobs(n_features=2, n_samples=100, centers=3, cluster_std=[0.3, 2, 4])