# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 18:46:38 2025

@author: Administrator
"""

a={'Python','Java'} #创建集合对象
x=set() #创建空集合
b_set=set(['data','information',2023,2.5]) 
del b_set #使用 del 语句删除集合
print(b_set)


b_set=set(['data','information',2023,2.5])
b_set.add('math') #add()函数用于增加集合元素
print(b_set) 

s={'Python','C','C++'} 
s.update({1,2,3},{'Wade','Nash'},{0,1,2}) 
print(s) #update()函数用于合并 3 个集合

s.remove('C') #删除某个元素，作用与 discard()函数一样
print(s) 

s.pop() #pop()函数用于随机删除集合中的一个元素
print(s) 

a=set('abcd') 
b=set('cdef') 
print(a-b) #差集

print(a|b) #并集

print(a&b) #交集

print('b' in a) #判断集合元素
