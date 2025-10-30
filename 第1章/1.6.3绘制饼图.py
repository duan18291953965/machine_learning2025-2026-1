# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:39:50 2025

@author: Administrator
"""

#使用 pie()函数绘制网络工程学院各党支部“学习强国”积分 10000 分以上的占比
import matplotlib 
import matplotlib.pyplot as plt 
matplotlib.rcParams['font.family']='SimSun' #导入中文字体
labels=['教工第一党支部','教工第二党支部','学生第一党支部','学生第二党支部'] 
X=[12,9,30,25] 
explode=(0.015,0.005,0.05,0.03) #设置扇形突出
plt.figure() 
plt.pie(X,explode=explode,labels=labels,autopct='%1.2f%%') 
plt.show()