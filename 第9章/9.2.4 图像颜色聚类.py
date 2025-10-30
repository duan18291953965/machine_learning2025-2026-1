# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 10:24:20 2024

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time
# 加载图像
china = load_sample_image("china.jpg")
china = np.array(china, dtype=np.float64) / 255 
plt.figure(1)
plt.axis("off")
plt.imshow(china)
# 生成样本
x = china.reshape(-1,china.shape[2])
# 设置颜色聚类数为2
n_colors = 2
# 颜色聚类
t0 = time()
KM = KMeans(n_clusters=n_colors).fit(x)
print(f'Time(all_samples): {time() - t0:0.2f}s.')
# 颜色分类
labels = KM.predict(x)
# 生成新图像
china_new_1 = KM.cluster_centers_[labels].reshape(china.shape[0],china.shape[1],-1)
# 显示新图像
plt.figure(2)
plt.axis("off")
plt.imshow(china_new_1)
# 将颜色聚类数设置为5并随机抽取指定样本点进行聚类
n_colors = 5
x_sample = shuffle(x, n_samples=500)
t0 = time()
KM = KMeans(n_clusters=n_colors).fit(x_sample)
print(f'Time (subset_samples): {time() - t0:0.2f}s.')
# 颜色分类
labels = KM.predict(x)
# 生成新图像
china_new_2 = KM.cluster_centers_[labels].reshape(china.shape[0],china.shape[1],-1)
# 显示新图像
plt.figure(3)
plt.axis("off")
plt.imshow(china_new_2)

