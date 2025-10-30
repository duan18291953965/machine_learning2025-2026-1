# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:30:04 2025

@author: Administrator
"""

class People: 
    name='' 
    age=0 
    __weight =0 #定义私有属性,私有属性在类外部无法直接进行访问
    def __init__(self,n,a,w): 
        self.name=n 
        self.age=a 
        self.__weight=w 
        def speak(self): 
            print('Name:%s;Age:%d'%(self.name, self.age)) 
#单继承示例
class Student(People): 
    grade='' 
    def __init__(self,n,a,w,g): 
        People.__init__(self,n,a,w) #调用父类的构造函数
        self.grade=g 
#重写父类的方法
    def speak(self): 
        print("Name:%s;Age:%d;Grade:%d"%(self.name,self.age,self.grade)) 
        s=Student('ken',10,30,3) 
        s.speak() #输出结果

#多继承示例
class Speaker(): #另一个类,多重继承之前的内容
    topic = '' 
    name = '' 
    def __init__(self,n,t): 
        self.name = n 
        self.topic = t 
        def speak(self): 
            print("I am %s, I like %s"%(self.name,self.topic)) 
#多重继承
class Sample(Speaker,Student): 
    a ='' 
    def __init__(self,n,a,w,g,t): 
        Student.__init__(self,n,a,w,g) 
        Speaker.__init__(self,n,t) 
        test = Sample("XiaoWang",25,80,4,"Python") 
        test.speak() #输出结果
