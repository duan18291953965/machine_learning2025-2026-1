# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:51:42 2024

@author: Administrator
"""

import sys
import os
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
# 导入数据标准化模块
from sklearn.preprocessing import StandardScaler
# 导入主成分分析
from sklearn.decomposition import PCA

# -------------------------- 路径处理核心代码 --------------------------
def get_resource_path(relative_path):
    """获取资源文件的绝对路径（兼容开发环境和PyInstaller打包环境）"""
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller打包后，文件会被提取到_MEIPASS临时目录
        base_path = sys._MEIPASS
    else:
        # 开发环境下，使用当前脚本所在目录
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# 加载数据（使用处理后的绝对路径）
csv_path = get_resource_path('Sleep_health_and_lifestyle_dataset.csv')
data = pd.read_csv(csv_path)
data = data.dropna()

# 数据预处理（修复筛选bug：替换前先筛选，或筛选时用数字）
# 先备份原始Sleep Disorder用于绘图，再进行替换
data['Sleep_Disorder_Original'] = data['Sleep Disorder']
data['BMI Category'].replace({'Normal':0,'Normal Weight':0,'Overweight':1,'Obese':2}, inplace=True)
data['Sleep Disorder'].replace({'None':0,'Sleep Apnea':1,'Insomnia':2}, inplace=True)

# -------------------------- 绘图部分（修复筛选条件） --------------------------
# 特征与目标之间的相关性
plt.figure(figsize=(10,5))
sns.barplot(x='Heart Rate', y='Sleep Disorder', data=data, orient='h', palette='rocket')
plt.figure(figsize=(10,5))
sns.barplot(x='Daily Steps', y='Sleep Disorder', data=data, orient='h', palette='rocket')

# 修复筛选条件：用原始字符串列筛选，而不是替换后的数字列
plt.figure(figsize=(15,10))
ax = sns.barplot(
    x='BMI Category',
    y='Sleep Duration',
    data=data[data['Sleep_Disorder_Original']!='None'],  # 使用原始列筛选
    hue='Sleep_Disorder_Original'  # 用原始列显示标签
)
plt.show()

# -------------------------- 后续分析代码不变 --------------------------
# 分离特征与类别标记
x = data.drop(['Sleep Disorder', 'Sleep_Disorder_Original'], axis=1)  # 删除备份列
y = data['Sleep Disorder']

# 特征相关性分析
corr = x[x.columns[0:]].corr()
plt.figure(figsize=(10,8))
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=False, annot=True, fmt='.1f'
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=30,
    horizontalalignment='right'
)
plt.show()

# 数据标准化
scaler = StandardScaler()
x = scaler.fit_transform(x)

# 绘制肘点图
K = np.arange(1,10)
Loss = []
for i in K:
    KM = KMeans(n_clusters=i, max_iter=100, random_state=42).fit(x)  # 添加random_state确保结果可复现
    Loss.append(KM.inertia_ / x.shape[0])
plt.figure()
plt.plot(range(1, 10), Loss, c='r', marker="o")
plt.xlabel('K')
plt.ylabel('Loss')
plt.plot(K, Loss, color='r', ls='--', marker='o')
plt.grid(True)
plt.show()

# 利用最优K值进行聚类
KM = KMeans(n_clusters=3, random_state=42)
KM.fit(x)
y_pred = KM.predict(x)
print('预测精度:', accuracy_score(y, y_pred))

# 降维并对降维数据聚类、绘制可视化图像
x_pca = PCA(n_components=2).fit_transform(x)
KM = KMeans(n_clusters=3, random_state=42)
KM.fit(x_pca)
y_pred = KM.predict(x_pca)
print('预测精度:', accuracy_score(y, y_pred))

# 绘制降维后的聚类结果（可选，增强可视化）
plt.figure(figsize=(10,6))
plt.scatter(x_pca[:,0], x_pca[:,1], c=y_pred, cmap='viridis', alpha=0.8)
plt.scatter(KM.cluster_centers_[:,0], KM.cluster_centers_[:,1], c='red', marker='x', s=200, linewidths=3)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-Means Clustering (PCA Reduced Data)')
plt.show()