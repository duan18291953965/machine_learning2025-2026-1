# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 18:45:11 2025

@author: Administrator
"""

dict={'Name':"xmj",'Age':17,'Class':'数据科学与大数据技术 1 班'} #创建字典
print("dict['Name']:",dict['Name']) #访问字典的值

dict={'Name':"Linda",'Age':20,'Class':'智能科学与技术 1 班'} #创建字典
dict['Age']=18 #更新键/值对
dict['School']="周口师范学院" #增加新的键/值对
del dict['Name'] #用 del 语句删除键是'Name'的元素
dict.clear() #用 clear()函数清空字典所有元素
del dict #删除字典

dict={'Name':"xmj",'Age':17,'Class':'网络工程 1 班'} #创建字典
print('Age' in dict) #判断 Age 是否在字典中

print(dict.values()) 

dict={'first':1, 'second':2} #创建字典
for kv in dict.items(): #以元组形式返回
 print(kv)
 
