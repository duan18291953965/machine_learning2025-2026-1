# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 18:55:47 2025

@author: Administrator
"""

class MyClass: 
 i = 12345 
 def f(self): 
     return 'hello world' 
x = MyClass() #实例化对象
print(x.i) #访问类的数据成员

print(x.f()) #访问类的成员方法

class Boy: 
 strName='zknu' 
 intAge=50 
 def __init__(self,name,age): 
     self.strName=name 
     self.intAge=age 
 def Intr(self): 
     print('My name:',self.strName) 
     print('My age:',self.intAge) 
Y=Boy('xiaowang',18) 
Y.Intr()


class JustCounter: 
 __secretCount = 0 #私有变量
 publicCount = 0 #公有变量
 def count(self): 
     self.__secretCount += 1 
     self.publicCount += 1 
     print (self.__secretCount) 
counter = JustCounter() 
counter.count() 
counter.count() 
print (counter.publicCount) 

print (counter.__secretCount) #提示错误，即实例不能访问私有变量
class Site: 
 def __init__(self, name): 
     self.name = name 
 def who(self): 
     print('name : ', self.name) 
 def __foo(self): #私有方法
     print('这是私有方法') 
 def foo(self): #公有方法
     print('这是公有方法') 
     self.__foo() 
x = Site('学习强国, www.xuexi.cn/') 
x.foo() 


