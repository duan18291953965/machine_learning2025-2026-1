# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 18:40:49 2025

@author: Administrator
"""

print(r'\\abcd\\')

print('Hello ! I %s a %s' % ('am', 'student!')) #占位符格式化
print('Hello ! I %(v1)s %(v2)s' % {'v1':'am','v2':'a student!'}) 
print('Hello ! I {} a {}'.format('am','student!')) #format()函数格式化