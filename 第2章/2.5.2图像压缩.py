# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 16:30:19 2024

@author: Administrator
"""

import numpy as np #导入科学计算库
from PIL import Image #导入图像处理库
import matplotlib.pyplot as plt #导入绘图模块
from sklearn.decomposition import PCA #导入主成分分析模块
#打开图像
im = Image.open('D:\教材源码\sample.jpg')     #需要调整路径
#显示图像
plt.figure(1)
plt.imshow(im) 
#显示图像信息
print('Image information',im.format, im.size, im.mode)
#提取并显示图像的红色通道
im_array = np.array(im) #转换为Numpy数组
im_red = im_array[:,:,0] #提取图像红色通道
plt.figure(2)
plt.imshow(im_red,cmap='gray') #显示图像红色通道
#图像主成分分析
#查看各主成分的方差值占总方差值的比例
pca = PCA(n_components=200) #将主成分设置为200
pca.fit(im_red) #主成分分析
#显示结果
plt.figure(3)
plt.plot(np.cumsum(pca.explained_variance_ratio_ *100))
plt.xlabel('Number of components')
plt.ylabel('Explained variance')
plt.grid()
#利用10个主成分进行图像压缩
pca_10 = PCA(n_components=10) #将主成分设置为10
im_red_pca_10 = pca_10.fit_transform(im_red) #主成分分析
im_red_inv_pca_10 = pca_10.inverse_transform(im_red_pca_10) #由主成分析生成图像
#显示压缩后的图像
plt.figure(4)
plt.imshow(im_red_inv_pca_10,cmap='gray') #显示图像
#显示主成分对应方差所占总方差的比例
#cumsum()函数用于数组的累计和
print('Explained_variance_ratio_10:', np.max(np.cumsum(pca_10.explained_variance_ratio_ *100)))
#利用50个主成分进行图像压缩
pca_50 = PCA(n_components=50) #将主成分设置为50
im_red_pca_50 = pca_50.fit_transform(im_red) #主成分分析
im_red_inv_pca_50 = pca_50.inverse_transform(im_red_pca_50) #由主成分析生成图像
#显示压缩后的图像
plt.figure(5)
plt.imshow(im_red_inv_pca_50,cmap='gray')
#显示主成分对应方差所占总方差的比例
print('Explained_variance_ratio_50:', np.max(np.cumsum(pca_50.explained_variance_ratio_ *100)))
