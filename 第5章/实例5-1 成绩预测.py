# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:45:04 2024

@author: Administrator
"""
from sklearn.naive_bayes import GaussianNB  # 导入高斯朴素贝叶斯库
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  # 导入特征编码库

# 初始数据
data = [
    ['Early', 'Late', 'High'],
    ['Early', 'Early', 'High'],
    ['Early', 'Late', 'Low'],
    ['Late', 'Early', 'Low'],
    ['Late', 'Late', 'High'],
    ['Early', 'Late', 'High'],
    ['Early', 'Early', 'High'],
    ['Late', 'Early', 'Low'],
    ['Late', 'Late', 'Low'],
    ['Early', 'Late', 'High']
]

# 分离特征与分类标记
x = [row[0:2] for row in data]
y = [row[2] for row in data]

print(x)
print(y)
#
# 分类标记转换（字符串转换为数值）
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
#
print("y的值 编码后修改为:",y)
#
# 特征编码（字符串转换为数值）
onehot_encoder = OneHotEncoder()
x = onehot_encoder.fit_transform(x).toarray()  # 转换为密集矩阵
print("x的值编码转换后",x)
#
# 构建朴素贝叶斯分类器
NB = GaussianNB()

# 训练朴素贝叶斯分类器
NB.fit(x, y)
#
# 新样本预测
new_data = [['Early', 'Late']]  # 使用列表的列表格式
onehot_new_data = onehot_encoder.transform(new_data).toarray()  # 转换新数据并转换为密集矩阵

# 预测"出勤早休息晚成绩为好或差"的概率
prob = NB.predict_proba(onehot_new_data)
# # 显示结果
print('成绩为好或差的概率:', prob)
#
# 预测"出勤早休息晚"相应成绩类别（0与1分别表示成绩好与差）
label = NB.predict(onehot_new_data)
# 显示结果
print('成绩类别:', label)

# 显示整体精度
print('预测精度:', NB.score(x, y))
