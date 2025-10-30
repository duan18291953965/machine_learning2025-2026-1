# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:41:29 2025

@author: Administrator
"""

import pandas as pd #导入 Pandas 库
import matplotlib.pyplot as plt 
df=pd.DataFrame({'男士':(300,800,450), '女士':(50,900,850)}) #创建 DataFrame 对象
df.plot(kind='bar',color=['red','cyan']) #绘制柱状图并设置柱体的颜色
plt.xticks([0,1,2], #设置 x 轴的刻度和文本
 ['从不戴头盔','有时忘记戴头盔','一直戴头盔'], 
 color='red', #字体颜色
 fontproperties='SimSun', #字体
 rotation=10) #旋转
plt.yticks(list(df['男士'].values)+list(df['女士'].values)) #设置 y 轴刻度
plt.ylabel('人数',fontproperties='SimSun',fontsize=10) 
plt.show()