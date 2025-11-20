# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:51:42 2024

@author: Administrator
"""

# 导入绘图模块
import matplotlib.pyplot as plt
import seaborn as sns
# 导入数据分析与计算模块
import numpy as np
import pandas as pd
# 导入K均值聚类模块
from sklearn.cluster import KMeans
# 导入度量模块
from sklearn.metrics import accuracy_score
#导入数据标准化模块
from sklearn.preprocessing import StandardScaler 
# 导入主成分分析
from sklearn.decomposition import PCA
#加载数据
data = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
data = data.dropna()
data['BMI Category'].replace({'Normal':0,'Normal Weight':0,'Overweight':1,'Obese':2},inplace = True)
data['Sleep Disorder'].replace({'None':0,'Sleep Apnea':1,'Insomnia':2},inplace = True)
# 特征与目标之间的相关性
plt.figure(figsize=(10,5))
sns.barplot(x='Heart Rate',y='Sleep Disorder',data = data, orient='h',palette = 'rocket')
plt.figure(figsize=(10,5))
sns.barplot(x='Daily Steps',y='Sleep Disorder',data = data, orient='h',palette = 'rocket')
plt.figure(figsize=(15,10))
ax = sns.barplot(x='BMI Category',y ='Sleep Duration',data = data[data['Sleep Disorder']!='None'],hue = 'Sleep Disorder')
plt.show()
# 分离特征与类别标记
x = data.drop('Sleep Disorder', axis= 1)
y = data['Sleep Disorder']
# 特征相关性分析
corr = x[x.columns[0:]].corr()
plt.figure(figsize=(10,8))
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=False, annot=True,fmt='.1f')
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=30,
    horizontalalignment='right'
)
plt.show()
#数据标准化
scaler = StandardScaler()
x = scaler.fit_transform(x)
#绘制肘点图
K = np.arange(1,10)
Loss = []
for i in K:
    KM = KMeans(n_clusters=i, max_iter=100).fit(x)
    Loss.append(KM.inertia_ / x.shape[0])
plt.figure()
plt.plot(range(1, 10), Loss, c='r', marker="o" )
plt.xlabel('K')
plt.ylabel('Loss')
plt.plot(K,Loss,color='r',ls='--',marker='o')
plt.grid(True)
plt.show()
# 利用最优K值进行聚类
KM = KMeans(n_clusters=3)
KM.fit(x)
y_pred = KM.predict(x)
print('预测精度:',accuracy_score(y,y_pred))
#降维并对降维数据聚类、绘制可视化图像
x_pca = PCA(n_components=2).fit_transform(x)
KM = KMeans(n_clusters=3)
KM.fit(x_pca)
y_pred = KM.predict(x_pca)
print('预测精度:',accuracy_score(y,y_pred))
