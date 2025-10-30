# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:49:05 2025

@author: Administrator
"""

from sklearn.preprocessing import OneHotEncoder 
X=[['BJ','BS'],['SH','SS'],['TJ','SS']] #共 3 个样本 2 个特征
enc = OneHotEncoder(drop='if_binary') 
enc.fit(X) 
Y = [['BJ','SS']] #测试样本
enc_Y = enc.transform(Y).toarray() 
print(enc_Y)