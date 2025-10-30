# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 19:00:13 2025

@author: Administrator
"""
import numpy as np
a= np.array([1,2,3]) #创建一个一维数组
a=np.array(((1,2,3),(4,5,6))) #创建一个二维数组
a=np.array([[[8,9],[8,8]],[4,8]]) #创建一个多维数组
print(a) 
#[[list([8, 9]) list([8, 8])] [4 8]] 
array=np.zeros(3) #使用 zeros()函数创建一维数组
print(array) 
#[0. 0. 0.] #3 个元素，值均为 0 
array=np.ones(4) #使用 ones()函数创建一维数组
print(array) 
#[1. 1. 1. 1.] #4 个元素，值均为 1 
array=np.empty(2) #使用 empty()函数创建一维数组
print(array) 
#[6.92552559e-312 0.00000000e+000] #2 个元素，值为随机数
a=np.arange(5) #等间隔创建数字数组
print(a) 
#[0 1 2 3 4] 
a=np.arange(1,10,2) #根据指定的区间与步长生成等差数据列
print(a) 
#[1 3 5 7 9]

b = np.array([4,5]) 
c = np.add(a,b) #数组相加
print(c) 
#[6 8] 
c = np.subtract(a,b) #数组相减
print(c) 
#[-2 -2] 

c= np.multiply(a,b) #数组相乘
print(c) 
#[8 15] 
c= np.dot(a,b) #矩阵相乘规则
print(c) 
#23 
c= np.divide (a,b) #数组相除
print(c) 
#[0.5 0.6] 
d = np.power(a,2) #求幂
print(d) 
#[4 9] 
d= np.mod(a,3) #求余数
print(d) 
#[2 0] 
d= np.sqrt(a) #求平方根
print(d) 
#[1.41421356 1.73205081] 
array1=np.array([[2,10],[1,5]]) 
array2=np.sum(array1) #元素之和
print(array2) 
#18 
array2=np.sum(array1,axis=0) #axis=0 表示对每一列（或称 0 维度）进行操作
print(array2) 
#[ 3 15] 
array2=np.sum(array1,axis=1) #axis=1 表示对每一行（或称 1 维度）进行操作
print(array2) 
#[ 12 6] 
array2=np.min(array1) #求所有元素中的最小值
print(array2) 
#1 
array2=np.min(array1, axis=0) #axis=0 表示对每一列进行操作
print(array2) 
#[1, 5] 
array2=np.min(array1, axis=1) #axis=1 表示对每一行进行操作
print(array2) 
#[2, 1] 
array2=np.max(array1) #max()是求最大值，操作方式与 min()类似
print(array2) 
#10 
array2=np.max(array1, axis=0) 
print(array2) 
#[2 10] 
array2=np.max(array1, axis=1) 
print(array2) 
#[10 5] 
array3=[0.45, 5.456, 4.2225, 9.655] 
array4=np.around(array3) #对值进行四舍五入取整
print(array4) 
#[0. 5. 4. 10.] 
array4=np.floor(array3) #向下取整
print(array4) 
#[0. 5. 4. 9.] 
array4=np. ceil(array3) #向上取整 
print(array4) 
#[ 1. 6. 5. 10.]


array1=np.array([1,2,3,4]) #创建一维数组
array2=array1.reshape(2,2) #使用 reshape()函数将原数组转换成二维数组
print(array2) 

array3=array1.reshape(2,3) #数量不相等
print(array3) #报错：cannot reshape array of size 4 into shape (2,3)

array1 = np.array([[1,2],[3,4]]) #创建二维数组
array2 = np.array([[7,8]]) #创建二维数组
array3=np.concatenate((array1,array2)) #将两个数组拼接成一个二维数组，拼接维度是 0 
print(array3) 

array4=np.concatenate((array1,array2.T),axis=1) #将两个数组拼接成一个二维数组，拼接维度是 1 
print(array4)

a = np.array([1,2]) 
b = np.array([7,8]) 
c=np.stack((a,b),axis=0) #在 0 维度进行堆叠
print(c) 

d=np.stack((a,b),axis=1) #在 1 维度进行堆叠
print(d) 

e=np.hstack((a,b)) #沿水平方向堆叠 
print(e) 

f=np.vstack((a,b)) #沿垂直方向堆叠
print(f)

a=np.array([[1,2,3],[4,5,6]]) #创建二维数组
b=np.append(a,[7,8,9]) #追加数组，并展平元素 
print(b) 

b=np.append(a,[[7,8,9]],axis=0) #在第 0 轴或第 1 维追加数组
print(b) 

b=np.append(a,[[1,1,1],[7,8,9]],axis=1) #在第 1 轴或第 2 维追加数组 
print(b) 

a = np.array([[5,2,7],[42,1,4]]) #创建二维数组
b = np.sort(a) #axis 的默认值为 1，表示按第 1 轴或第 2 维排序数组
print(b) 

c=np.sort(a,axis=0) #axis 设置为 0，表示依据第 0 轴或第 1 维排序数组
print(c) 

b=np.argmax(a,axis=1) #按行求最大元素索引
print(b) 

b=np.argmin(a,axis=0) #按列求最小元素索引
print(b) 

a1 = np.array([[1, 2, 1, 2], [3, 4, 4, 4], [1, 2, 2, 2], [3, 4, 4, 4], [2, 4, 2, 4]]) 
u_a11 = np.unique(a1, axis=0) # 沿指定的轴方向去除数值完全相同的元素
print(u_a11) #去除值完全相同的行
 
u_a12 = np.unique(a1, axis=1) 
print(u_a12) # 去除值完全相同的列


a=np.array([2,38,3,4]) #创建一维数组
b=np.where(a>3) #符合条件的元素索引 
print(b) 

a=np.array([[5,2],[42,1]]) #创建二维数组
b=np.where(a>4) 
print(b) #返回两个数组，第一个数组从行开始描述，第二个数组从列开始描述