# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 18:43:43 2025

@author: Administrator
"""

tup=() #创建空元组
tup1=('中国',) #创建只包含一个元素的元组
tup2=('中国','美国',1997,2000) 
print(tup2[0]) #使用索引来访问元组中的值

del tup2 #使用 del 语句删除整个元组
print(tup2) 

tuple = ( 'Python', 999 , 9.96, 'Java', 888 ) #创建元组
tinytuple = (123, 456) #创建元组
y=max(tinytuple) #max()返回元组元素最大值
print(y) 

print(tuple[1:3]) #元组切片

print(tinytuple * 2) #重复元组

print(tuple + tinytuple) #连接元组

tuple=(1,2,3) #创建元组
list1=list(tuple) #将元组转换成列表
print(list1) 

num=[1,2,3] #创建列表
print(tuple(num)) #列表转换成元组

str1="I love China!" #创建字符串
list1=str1.split(" ") #字符串转换成列表
print(list1) 

list1=['I', 'love', 'China!'] #创建列表
str1=" ".join(list1) #列表转换成字符串
print(str1) 



