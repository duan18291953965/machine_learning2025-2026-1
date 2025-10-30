# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:45:37 2025

@author: Administrator
"""

from sklearn.preprocessing import OrdinalEncoder 
X = [['BJ','BS'],
     ['SH','SS'],
     ['TJ','SS'],
     ['XA','ZL']
     ] #共 3 个样本 2 个特征


# B J   0
# SH 1
# tj 2

#bs  0   ss 1
enc = OrdinalEncoder() #定义特征编码对象
enc.fit(X) #特征编码
Y = [['BJ','SS']] #测试样本
Z= [['TJ','ZL']]
enc_Y = enc.transform(Y)
enc_Z = enc.transform(Z)
print(enc_Y)
print("Z的编码是：")
print(enc_Z)