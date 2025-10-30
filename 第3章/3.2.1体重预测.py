# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:36:31 2024

@author: Administrator
"""

from sklearn.linear_model import LinearRegression #安装scikit-learn
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
#训练数据
X = [[0.86],[0.96],[1.12],[1.35],[1.55],[1.63],[1.71],[1.85]]
Y = [[12],[15],[20],[35],[48],[51],[59],[75]]
#线性回归
model = LinearRegression(fit_intercept=True)# y=kx+b   b计算出
model.fit(X,Y)
k = model.coef_[0][0] #斜率
b = model.intercept_[0]#截距
print(k)
print(b)

#显示结果
plt.figure(1)
plt.xlabel('Height(m)'),plt.ylabel('Weight(kg)')
plt.scatter(X, Y, color='white', edgecolors='k', label='Data Points')#散点图
plt.plot(X,model.predict(X), color='red', linewidth=1, label='Fitted Line')#预测直线
plt.legend(loc='upper left') #标签显示
plt.grid(True); plt.show()
#模型测试
x2 = [[0.83], [1.08], [1.26], [1.51], [1.6], [1.67], [1.75], [1.90]]
y2 = [[11], [17], [27], [41], [50], [64], [66], [89]]
fitted_y = k * np.array(x2) + b
plt.figure(2)
plt.xlabel('Height(m)'),plt.ylabel('Weight(kg)')
plt.plot(x2,fitted_y, color='red', linewidth=1, label='Line Model')#已拟合直线
plt.scatter(x2, fitted_y, color='green', label='Predicted Points')#测试点
#体重高于预测值用白色正方形显示，否则用黑色正方形显示
cm_pt = mpl.colors.ListedColormap(['w', 'k']) #设置颜色映射
yn = fitted_y - y2; yn[yn > 0] = 1; yn[yn <= 0] = 0 #比较体重与预测值
plt.scatter(x2, y2, c=yn, cmap=cm_pt, marker='s', edgecolors='k', label='Test Points')
plt.legend(loc='upper left') #标签显示
plt.grid(True); plt.show()
