# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 18:51:04 2025

@author: Administrator
"""

def printme(str): 
 print (str) 
 return 
printme() #参数调用错误，将显示错误提示！
printme(str="python")

def printinfo( name, age ): 
 #输出任何传入的字符串
 print ("名字: ", name) 
 print ("年龄: ", age) 
 return 
printinfo( age=50, name="张三" )

def printinfo( name, age = 35 ): #age 的默认值为 35 
 print ("名字: ", name) 
 print ("年龄: ", age) 
 return 
printinfo( name="李四" )

def printinfo( arg1, *vartuple ): 
 print (arg1) 
 print (vartuple) 
printinfo( 70, 60, 50 ) #*参数放在元组中

printinfo( 10 ) #没指定*参数，则视为空元组
 
def printinfo( arg1, **vardict ): 
 print (arg1) 
 print (vardict) 
printinfo(1, a=2,b=3) #**参数放在字典中


sum = lambda arg1, arg2: arg1 + arg2 # 求和函数
print ("相加后的值为: ", sum( 10, 20 )) # 调用 sum()函数

#计算任意数的阶乘
def func(num): 
 count=num 
 if count==1: 
     result=1 
 else: 
     result=func(count-1)*count #函数递归调用
 return result 
print(func(5))


total = 0 #全局变量
def sum( arg1, arg2 ): 
 total = arg1 + arg2 #此处 total 为局部变量
 print ("函数内是局部变量: ", total) #此处 total 为 30 
 return total 
sum( 10, 20 ) 
print ("函数外是全局变量: ", total) #此处 total 为 0


x = 99 
def func(): 
     global x #修改全局作用域
     x = 88 
func() 
print(x) 

def func(): 
 count = 1 
 def foo(): 
     nonlocal count #修改嵌套作用域
     count = 12 
     foo() 
     print(count) 
func()