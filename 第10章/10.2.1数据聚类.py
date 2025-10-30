# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# 导入绘图库
import matplotlib.pyplot as plt 
import seaborn as sns; sns.set() 
from matplotlib.patches import Ellipse 
#导入科学计算库
import numpy as np 
from scipy.spatial.distance import cdist 
from sklearn.mixture import GaussianMixture as GMM #导入高斯混合模型模块
from sklearn.cluster import KMeans #导入 K 均值聚类模块
# 构造球形数据
from sklearn.datasets import make_blobs 
from sklearn.datasets import make_moons 
x, y = make_blobs(n_samples=200, centers=4, cluster_std=[2.0,1.0,0.9,1.5], 
random_state=1) 
# 利用 K 均值聚类算法进行聚类
KM = KMeans(n_clusters=4, max_iter=100).fit(x) 
km_center = KM.cluster_centers_ 
labels = KM.predict(x) 
plt.figure(1) 
plt.axis('equal') 
plt.scatter(x[:, 0], x[:, 1], c=labels, s=20, cmap='gist_rainbow', marker = 
'o',linewidths=1, edgecolors='k') 
km_radius = [cdist(x[labels == i], [c]).max() for i, c in enumerate(km_center)] 
for c, r in zip(km_center, km_radius): 
    plt.gca().add_patch(plt.Circle(c, r, fc='r', lw=3, alpha=0.2, zorder=1))
plt.xlabel('x1')
plt.ylabel('x2')
plt.show() 
# 构造非球形数据
x, y = make_moons(n_samples=200,noise=0.1) 
# 利用 K 均值聚类算法进行聚类
KM = KMeans(n_clusters=4, max_iter=100).fit(x) 
km_center = KM.cluster_centers_ 
labels = KM.predict(x) 
plt.figure(2) 
plt.axis('equal') 
plt.scatter(x[:, 0], x[:, 1], c=labels, s=20, cmap='gist_rainbow', marker = 
'o',linewidths=1, edgecolors='k') 
km_radius = [cdist(x[labels == i], [c]).max() for i, c in enumerate(km_center)] 
for c, r in zip(km_center, km_radius):
    plt.gca().add_patch(plt.Circle(c, r, fc='r', lw=3, alpha=0.2, zorder=1)) 
plt.xlabel('x1')
plt.ylabel('x2')
plt.show() 
# 利用高斯混合模型进行聚类
#更改分量数生成不同的结果
GM = GMM(n_components=10, covariance_type='full', random_state=0).fit(x) 
labels = GM.predict(x) 
plt.figure(3) 
plt.axis('equal') 
plt.scatter(x[:, 0], x[:, 1], c=labels, s=20, cmap='gist_rainbow', marker = 
'o',linewidths=1, edgecolors='k') 
w_factor = 0.2 / GM.weights_.max() 
for pos, covar, w in zip(GM.means_, GM.covariances_, GM.weights_): 
    if covar.shape == (2, 2): 
        U, s, Vt = np.linalg.svd(covar) 
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0])) 
        width, height = 2 * np.sqrt(s) 
    else: 
        angle = 0 
        width, height = 2 * np.sqrt(covar) 
    for nsig in range(1, 4): 
        plt.gca().add_patch(Ellipse(pos, nsig * width, nsig * height, angle, alpha= 
w * w_factor)) 
plt.xlabel('x1')
plt.ylabel('x2')
plt.show() 
# 利用高斯混合模型生成新数据
x_new,y_new = GM.sample(200) 
plt.figure(4) 
plt.scatter(x[:, 0], x[:, 1], c=labels, s=20, cmap='gist_rainbow', marker = 
'o',linewidths=1, edgecolors='k') 
plt.xlabel('x1')
plt.ylabel('x2')
plt.show() 
# 最优分量数的求取
n_components = np.arange(1, 21) 
GMs = [GMM(n, covariance_type='full', random_state=0).fit(x) for n in n_components] 
plt.figure(5) 
plt.plot(n_components, [m.bic(x) for m in GMs], label='BIC') 
plt.plot(n_components, [m.aic(x) for m in GMs], label='AIC') 
plt.legend(loc='best') 
plt.xlabel('n_components') 
plt.ylabel('BIC and AIC') 
plt.grid(True) 
plt.show()