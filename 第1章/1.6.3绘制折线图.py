# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:37:23 2025

@author: Administrator
"""

import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

x = np.linspace(-np.pi,np.pi,50) #设置自变量
y = np.sin(x) #求取因变量
plt.figure() 
plt.plot(x, y, color='r', linestyle=':') #设置颜色与线型
plt.xlabel('X') #设置横轴标签
plt.ylabel('Y的坐标') #设置纵轴标签
#plt.grid(True) #显示网格线
#plt.title('sin function',fontproperties='SimSun',fontsize=24) #设置标题文本、字体、字号
plt.show() #显示绘制的结果图像