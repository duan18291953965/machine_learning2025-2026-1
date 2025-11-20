# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:11:01 2024

@author: Administrator
"""

from sklearn.feature_extraction.text import TfidfTransformer #导入TF-IDF特征处理模块
from sklearn.cluster import KMeans #导入K均值聚类库
from sklearn.feature_extraction.text import CountVectorizer #导入
from sklearn.decomposition import PCA # 导入主成分分析模块
#文本数据
wordlist = [
    'Got it!',
    'What are you going to do?',
    'An idle youth,a needy age.',
    'He has a large income.',
    'How blue the sky is!',
    'What is on the schedule for today?',
    'You look beautiful tonight.',
    'I promise.',
    'How great you are!',
    'I got sick and tired of hotels.',
    'I am sorry I took so long to reply.',
    'I hope everything is all right.',
    'When are you free?',
    'What are you in the mood for?']
#文本转换为词频矩阵
matrix = CountVectorizer()
#计算词语出现次数
count = matrix.fit_transform(wordlist)
#统计单词
word = matrix.get_feature_names_out()
print('所有单词:',word)
#将词频矩阵转换为TF-IDF值
transformer = TfidfTransformer()
idf = transformer.fit_transform(count)
#查看TF-IDF权重
x = idf.toarray()
print('样本基本信息:',x.shape)
#主成分分析
x_pca = PCA(n_components=0.98).fit_transform(x)
print('样本基本信息(PCA):',x_pca.shape)
#利用KMeans算法进行聚类
km = KMeans(n_clusters=3, random_state=0).fit(x_pca)
#输出聚类结果
print ('聚类结果:',km.labels_)
#输出每类对应的文本
max_centroid = 0
max_cluster_id = 0
cluster_list = []
for i in range(3):
    members_list = []
    for j in range(0, len(km.labels_)):
        if km.labels_[j] == i:
            members_list.append(j)
    cluster_list.append(members_list)
for i in range(0,len(cluster_list)):
    print ('第' + str(i+1) + '类:')
    for j in range(0,len(cluster_list[i])):
        ix = cluster_list[i][j]
        print (wordlist[ix])

