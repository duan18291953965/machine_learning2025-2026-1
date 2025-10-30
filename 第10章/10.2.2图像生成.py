# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:19:41 2024

@author: Administrator
"""

import numpy as np #导入科学计算库
from torch.utils.data import DataLoader #导入图像加载模块
from torchvision import datasets, transforms #导入数据与预处理模块
import matplotlib.pyplot as plt #导入绘图库
from sklearn.decomposition import PCA #导入主成分分析模块
from sklearn.mixture import GaussianMixture #导入高斯混合模型模块
# 加载图像数据
batch_size = 16 
mnist_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms. 
ToTensor(), download=True) 
mnist_data = DataLoader(dataset=mnist_dataset, batch_size=batch_size, shuffle=True) 
# 显示图像
image_set=enumerate(mnist_data) 
idx,(images,targets)=next(image_set) 
fig, axes = plt.subplots(nrows=4, ncols=4) 
for ax, im in zip(axes.ravel(),images): 
    im = im.view(-1,28) 
    ax.imshow(im,cmap='gray') 
plt.tight_layout() 
plt.show()
print('数据基本信息: ',images.shape)
# 主成分分析
image = images.view(images.size(0), -1) 
pca = PCA(0.9, whiten=True) 
image_pca = pca.fit_transform(image) 
print('数据基本信息(PCA): ',image_pca.shape)
# 构建高斯混合模型
GM = GaussianMixture(5, covariance_type='full', random_state=2) 
GM.fit(image_pca) 
# 产生新图像
images_new = GM.sample(batch_size) 
images_new = pca.inverse_transform(images_new[0]) 
# 显示新图像
im_new = np.resize(images_new, (images_new.shape[0], images.shape[2], images.shape[3])) 
fig, axes = plt.subplots(nrows=4, ncols=4) 
for ax, im in zip(axes.ravel(),im_new): 
    ax.imshow(im,cmap='gray') 
plt.tight_layout() 
plt.show()