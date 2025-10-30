# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 18:48:42 2025

@author: Administrator
"""

score=95 
if score>100 or score<0: 
 print("非法成绩") 
elif score>=60 and score<70: 
 print("合格") 
elif score>=70 and score<90: 
 print("良好") 
elif score>=90 and score<=100: 
 print("优秀") 
else: 
 print("不及格")
 
 
if a<b: #pass 语句用在选择结构中
 pass 
else: 
 z=a 
class A: #pass 语句用在类的定义中
 pass 
def demo(): #pass 语句用在函数的定义中
 pass 


count = 0 
while count < 5: 
 print('The count is:',count) 
 count = count + 1 
print("Good bye!")


fruits = ['banana','apple','mango'] #定义一个列表
for i in range(len(fruits)): #循环变量为索引
 print('当前水果:',fruits[i]) 
print("Good bye!")

var = 3 
while var > 0: 
 var = var -1 
 if var == 1: 
     continue 
 print("当前变量值：",var) 
print("Good bye!")

for i in range(1,10): 
 for j in range(1,i+1): 
     print(i,'*',j,'=',i*j,'\t',end=" ") #end=" "的作用是不换行
     print("") #仅起换行作用
