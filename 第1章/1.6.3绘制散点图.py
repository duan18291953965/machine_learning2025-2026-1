# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:39:07 2025

@author: Administrator
"""

import numpy as np 
import matplotlib.pyplot as plt 
x=np.linspace(10,100,50) #生成 x 轴数据
y=np.random.rand(50) #生成 y 轴数据
plt.xlabel('X') #设置横轴标签
plt.ylabel('Y') #设置纵轴标签
#散点大小与位置有关，设置颜色、形状以及透明度等参数
plt.scatter(x, y, c='r',s=x*y,alpha=0.8,marker='*')
plt.show()