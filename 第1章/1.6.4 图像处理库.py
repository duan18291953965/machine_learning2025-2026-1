# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 19:22:55 2025

@author: Administrator
"""

from PIL import Image 

im = Image.open('zknu.jpg') #加载图像
im.show() #显示图像

#plt.imshow(im) 
#plt.axis('off') #不显示坐标轴

print(im.format, im.size, im.mode)

im.save('002.png','PNG')

im_array = np.array(im) #转换为 NumPy 数组
im_new = Image.fromarray(im_array) # NumPy 数组转换为图像

im_1=plt.imshow(im_array[:,:,0]) #显示第 1 个通道

im_gray=im.convert('L') #转换为灰度图
im_bin=im.convert('1') #转换为二值图
#通过设定阈值进行二值化
threshold = 128 #设置阈值
im_bin_new = im_gray.point(lambda x: 0 if x < threshold else 255, '1') #转换为二值图
im_bin.show() #显示图像

rc=[100,200] #指定行列
RGB=im_array[rc[0],rc[1],:] #获取像素 RGB 值
print(RGB) 

RGB=im.getpixel((200,100)) #使用 getpixel()函数获取指定像素的 RGB 值
print(RGB) 


im_array[i,:] = im_array[j,:] #将第 j 行像素值赋给第 i 行
im_array[:,i] = 255 #将第 i 列的所有像素值设为 255 
im_array[10:20,30:40] #获取第 10～20 行与第 30～40 列像素值（不含第 20 行与第 40 列）
im_array[:,-1] #获取最后 1 列像素值（负序号表示逆向计数）

rc=[100,200] #指定像素行列
im_array[rc[0],rc[1],:]=[255,0,0] #将指定像素颜色修改为红色
im.putpixel((200,100),(255,0,0)) #采用 putpixel()函数进行修改

im_small=im.resize((128,128)) #修改图像尺寸
im_rotate= im.rotate(45) #旋转 45°
box=(300,200,600,300) #指定区域
sub_im=im.crop(box) #截取区域

im_filter=im.filter(ImageFilter.CONTOUR) #轮廓
im_filter = im.filter(ImageFilter.GaussianBlur(radius=2)) #模糊
im_filter = im.filter(ImageFilter.EMBOSS) #浮雕

plt.imshow(im) 
x =[300,300,400,400] 
y =[200,300,200,300] 
plt.plot(x,y,'ro') #红色圆点标记
plt.plot([x[0],x[3]],[y[0],y[3]],'b') #连接第 1 点与第 4 点
plt.plot([x[1],x[2]],[y[1],y[2]],'g') #连接第 2 点与第 3 点
plt.show() 


import matplotlib.pylab as mp 
plt.imshow(im) 
xy=mp.ginput(3) #在图像中单击 3 次可将单击处坐标保存至 xy 
print(xy) #输出坐标


import torchvision.transforms as T 

#导入图像处理相关库并加载示例图像
import numpy as np #导入科学计算库
import matplotlib.pyplot as plt #导入绘图库
from PIL import Image 
import torchvision.transforms as T 
#加载与打开图像
im = Image.open('sample.jpg') 
im.show()

transform = T.CenterCrop(100) #定义图像剪裁对象
sub_im = transform(im) #对图像进行剪裁
plt.imshow(np.array(sub_im))

transform=T.ColorJitter(brightness=0.8, contrast=0.3, saturation=0.9, hue=0.2) #定义图像操作
sub_im = transform(im) 
plt.imshow(sub_im)

transform=T.FiveCrop(100) 
sub_ims=transform(im) 
for sub_im in sub_ims: 
 plt.figure() 
 plt.imshow(sub_im)
 
transform=T.Grayscale(num_output_channels=3) #参数可设置为 1 或 3（通道数）
im_gray=transform(im) 
plt.imshow(im_gray, cmap ='gray')

transform = T.Pad(padding=(2,4,6,8), fill=(255, 0, 0), padding_mode='constant') 
result = transform(im) 
plt.imshow(np.array(result))

transform = T.RandomAffine(degrees=(-30,30), translate=None, scale=None, shear=30, 
resample=Image.BILINEAR, fillcolor=(255,0,0)) 
im_new = transform(im) 
plt.imshow(im_new)

transform = T.RandomCrop((100,200)) 
sub_im = transform(im) 
plt.imshow(sub_im)

transform = T.RandomGrayscale(p=0.1) 
sub_im = transform(im) 
plt.imshow(sub_im)

transform = T.RandomHorizontalFlip(p=0.9) #水平翻转
im_new = transform(im) 
plt.imshow(im_new) 
transform = T.RandomVerticalFlip(p=0.9) #垂直翻转
im_new = transform(im) 
plt.imshow(im_new)

transform = T.Resize((20,30), interpolation=Image.BILINEAR) 
im_new = transform(im) 
plt.imshow(im_new)

transforms = [ 
 T.CenterCrop(160), 
 T.Pad(padding=(2,4,6,8), fill=(255, 255, 255), padding_mode='constant') 
] 
transform = T.RandomApply(transforms, p=0.9) 
im_new = transform(im) 
plt.imshow(im_new)


