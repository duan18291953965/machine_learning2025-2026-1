# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 18:41:44 2025

@author: Administrator
"""

list_example=['xiaowang','xiaozhang','xiaohua'] 
print(list_example[0]) #访问列表中的第 1 个元素

print(list_example[1]) #访问列表中的第 2 个元素

list1=['中国','美国',1997,2000] 
del list1[2] #删除索引为 2 的元素
list1.remove('美国') #删除值为“美国”的元素
list1.pop(1) #删除索引为 1 的元素
print(list1)


list1=['中国','美国'] 
list1.append(2003) #在尾部追加元素
list1.insert(0,2023) #在指定位置插入元素
list1.extend([2022,2021]) #在尾部追加多个元素
print(list1) 


list2=[["CPU","内存"],["硬盘","声卡"]] #定义一个二维列表
print(list2[0][1]) #以“列表名[索引 1][索引 2]”的方式获取元素

list3=[1,2] 
list4=[3,4] 
print(list3+list4) #组合列表

print(list3*2) #重复列表
