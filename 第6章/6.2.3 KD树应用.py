# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:46:21 2024

@author: Administrator
"""

#导入绘图库
from matplotlib import pyplot as plt 
from matplotlib.patches import Circle
from sklearn.neighbors import KDTree,KNeighborsClassifier #导入KD树模块
from sklearn.datasets import make_blobs #导入make_blobs数据库
# 加载数据
x, y = make_blobs(n_samples = 50, centers=4, cluster_std=[2.6,1.2,1.5,3], random_state=1)
# 构造KD树
tree = KDTree(x, leaf_size = 20, metric='euclidean')
# 指定样本点
point = [x[20]]
# 查找指定样本点的近邻点(数量)
dis_, ix_ = tree.query(point, k=5, return_distance=True)
print('近邻点序号(KD树-数量):',ix_)
print('近邻点距离与样本点之间的距离(KD树-距离):',dis_)
# 查找指定样本点的近邻点(半径)
ix, dis = tree.query_radius(point, r=1.5, return_distance=True)
print('近邻点序号(KD树-半径):',ix)
print('近邻点距离与样本点之间的距离(KD树-半径):',dis)

# KD树近邻点可视化
plt.figure(2)
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(x[:, 0], x[:, 1], c=y, s=100, cmap='gist_rainbow', marker = 'o',linewidths=1, edgecolors='k')
cir = Circle(point[0], 1.5, color='r', fill=False)
for i in ix_[0]:
    plt.plot([point[0][0], x[i][0]], [point[0][1], x[i][1]], 'k-.', linewidth=1.5)
plt.gca().add_patch(cir)
plt.show()



# 利用K近邻分类器生成近邻点
# 构建K近邻分类器 
KNN = KNeighborsClassifier(n_neighbors=5) #指定近邻数
# 训练K近邻分类器
KNN.fit(x, y)
# 指定样本点
point = [x[20]]
# 指定样本点的近邻点
dis_, ix_ = KNN.kneighbors(point, return_distance=True)
# 显示近邻点的基本信息
print('近邻点序号(KNN-数量):',ix_)
print('近邻点距离与样本点之间的距离(KNN-距离):',dis_)


#导入绘图库
from matplotlib import pyplot as plt 
from matplotlib.patches import Circle 
from sklearn.neighbors import KDTree #导入 KD 树模块
from sklearn.datasets import make_blobs #导入 make_blobs 数据集
# 加载数据
x, y = make_blobs(n_samples = 50, centers=4, cluster_std=[2.6,1.2,1.5,3], random_state=1) 
# 构造 KD 树
tree = KDTree(x, leaf_size = 20, metric='euclidean') 
# 指定样本点
point = [x[20]] 
# 查找指定样本点的近邻点（数量）
dis_, ix_ = tree.query(point, k=5, return_distance=True) 
print('近邻点序号(KD 树-数量):',ix_) 
print('指定样本点与其近邻点之间的距离(KD 树-距离):',dis_) 
# 查找指定样本点的近邻点（半径）
ix, dis = tree.query_radius(point, r=1.5, return_distance=True) 
print('近邻点序号(KD 树-半径):',ix) 
print('指定样本点与其近邻点之间的距离(KD 树-半径):',dis) 
# 近邻点可视化
plt.figure(2) 
plt.xlabel('X') 
plt.ylabel('Y') 
plt.scatter(x[:, 0], x[:, 1], c=y, s=100, cmap='gist_rainbow', marker = 'o',linewidths=1, 
edgecolors='k') 
cir = Circle(point[0], 1.5, color='r', fill=False) 
for i in ix_[0]: 
 plt.plot([point[0][0], x[i][0]], [point[0][1], x[i][1]], 'k-.', linewidth=1.5) 
plt.gca().add_patch(cir) 
plt.show() 
# 利用 K 近邻分类器生成近邻点
# 构建 K 近邻分类器 
KNN = KNeighborsClassifier(n_neighbors=5) #指定近邻数
# 训练 K 近邻分类器
KNN.fit(x, y) 
# 指定样本点
point = [x[20]] 
# 指定样本点的近邻点
dis_, ix_ = KNN.kneighbors(point, return_distance=True) 
# 显示近邻点的基本信息
print('近邻点序号(KNN-数量):',ix_) 
print('指定样本点与其近邻点之间的距离(KNN-距离):',dis_)



