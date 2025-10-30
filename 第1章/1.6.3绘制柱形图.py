# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:40:09 2025

@author: Administrator
"""

#使用 bar()函数绘制柱状图，实现水平叠加与垂直叠加功能
import numpy as np 
import matplotlib.pyplot as plt 
index = np.arange(4) #生成一维数组
B1=[12,24,8,25] 
B2=[8,20,15,38] 
labels=[B1,B2] 
bar_width=0.2 
plt.bar(index,B1,bar_width,color='c') #绘制柱状图，并设置宽度、颜色等参数
plt.bar(index+bar_width,B2,bar_width,color='r') #水平叠加的 B2 柱状图
#plt.bar(index,B2,bar_width,color='r',bottom=B1) #垂直叠加的 B2 柱状图
plt.legend(labels,loc='upper left',ncol=1) #图例设置（左上角、单例）
plt.show()