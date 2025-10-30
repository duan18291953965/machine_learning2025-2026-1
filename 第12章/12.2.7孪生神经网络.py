# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:17:48 2024

@author: Administrator
"""

#导入 PyTorch 框架库及 Torchvision 库
import torch 
from torch import nn, optim 
from torch.utils.data import DataLoader,Dataset 
from torchvision import datasets, transforms 
import torch.nn.functional as F 
#导入绘图库
import matplotlib.pyplot as plt 
#导入图像处理库的 Image 模块
from PIL import Image 
import PIL 
#导入科学计算库
import numpy as np 
#导入文件处理库
import os 
import glob 
#定义图像显示函数
def show_image(im): 
 plt.figure() 
 im = im/2 + 0.5 
 im = im.numpy() 
 plt.imshow(np.transpose(im,(1,2,0)),cmap='gray') 
 plt.show() 
#定义数据集类
class MyDataSet(Dataset): 
    def __init__(self,im_dir,transform=None,yn_invert=True): 
        self.im_dir = im_dir #存放图像的文件夹
        self.transform = transform #预处理
        self.yn_invert = yn_invert #是否进行通道反转
        self.image_list = glob.glob(self.im_dir + '/*/*.jpg') #图像列表
    def __getitem__(self, *args): 
        label_yn = np.random.randint(2) #两幅图像若相似为 1，若不相似为 0 
        im_A_path = np.random.choice(self.image_list) #随机选择 1 幅图像
        im_A_label = int(os.path.split(im_A_path)[0].split('\\')[-1]) #图像真实类别
        if label_yn: #抽取与当前图像属于同一个类别的图像
            while True: 
                im_B_path = np.random.choice(self.image_list) #随机选择图像
                #图像真实类别
                im_B_label = int(os.path.split(im_B_path)[0].split('\\')[-1]) 
                if im_A_label == im_B_label: #若类别相同则终止
                    break 
        else: 
            while True: 
                im_B_path = np.random.choice(self.image_list) #随机选择图像
                #图像真实类别
                im_B_label = int(os.path.split(im_B_path)[0].split('\\')[-1]) 
                if im_A_label != im_B_label: #若类别不相同则终止
                    break 
 #读取图像
        im_A = Image.open(im_A_path) 
        im_B = Image.open(im_B_path) 
 #判断是否进行通道反转
        if self.yn_invert: 
            im_A = PIL.ImageOps.invert(im_A) 
            im_B = PIL.ImageOps.invert(im_B) 
 #判断是否进行预处理
        if self.transform is not None: 
            im_A = self.transform(im_A) 
            im_B = self.transform(im_B) 
            return im_A, im_B, label_yn #返回两幅图像与相似标记
    def __len__(self): 
        return len(self.image_list) 
#定义预处理操作
transform = transforms.Compose([ 
 transforms.Grayscale(num_output_channels=1), #转成单通道 
 transforms.ToTensor(), #转换为张量
 transforms.Normalize((0.5,), (0.5,)), #归一化 
 ]) 
#加载数据 
train_dir = "./DATA/CIFAR/train/" #训练集文件夹 
train_dataset = MyDataSet(im_dir=train_dir,transform=transform,yn_invert=False) 
train_dataloader = DataLoader(train_dataset,shuffle=True,batch_size=32) 
test_dir = "./DATA/CIFAR/test/" #测试集文件夹 
test_dataset = MyDataSet(im_dir=test_dir,transform=transform,yn_invert=False) 
test_dataloader = DataLoader(test_dataset,shuffle=True,batch_size=32) 
#构造孪生神经网络
class SiameseNetwork(nn.Module):
    def __init__(self): 
        super().__init__() 
        self.cnn = nn.Sequential( 
            nn.ReflectionPad2d(1), 
            nn.Conv2d(1, 5, 3, 1), 
            nn.ReLU(), 
            nn.BatchNorm2d(5), 
            nn.ReflectionPad2d(1), 
            nn.Conv2d(5, 10, 3, 1), 
            nn.ReLU(), 
            nn.BatchNorm2d(10), 
            nn.ReflectionPad2d(1),
            nn.Conv2d(10, 20, 3, 1), 
            nn.ReLU(inplace=True), 
            nn.BatchNorm2d(20), 
 ) 
        self.fc = nn.Sequential( 
            nn.Linear(32*32*20, 100), 
            nn.ReLU(), 
            nn.Linear(100, 50), 
            nn.ReLU(), 
            nn.Linear(50, 5)) 
    def forward_once(self, x): 
        y = self.cnn(x) 
        y = y.view(y.size()[0], -1) 
        y = self.fc(y) 
        return y 
    def forward(self, x1, x2): 
        y1 = self.forward_once(x1) 
        y2 = self.forward_once(x2) 
        return y1, y2 
net = SiameseNetwork() #定义模型
#定义对比损失函数
class ContrastiveLoss(torch.nn.Module): 
    def __init__(self, margin=2.0): 
        super(ContrastiveLoss, self).__init__() 
        self.margin = margin 
    def forward(self, x1, x2, label): 
        euclidean_distance = F.pairwise_distance(x1, x2, keepdim = True) 
        loss = torch.mean((1-label) * torch.pow(euclidean_distance, 2) + (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)) 
        return loss 
loss = ContrastiveLoss() 
#定义优化器
optimizer = optim.Adam(net.parameters(), lr = 0.001) #优化器
#训练模型
T = 10 
for epoch in range(T): 
    loss_ = 0.0 #累积训练误差
    for i, data in enumerate(train_dataloader, 1): 
        im_1, im_2, label = data 
        output1, output2 = net(im_1, im_2) 
        L = loss(output1, output2, label) 
        optimizer.zero_grad() 
        L.backward() 
        optimizer.step() 
        loss_ += L.data.numpy() #误差累积
 #显示误差变化
    if (epoch==0) | ((epoch+1) % 2 == 0):
        print('Epoch:[{}/{}], Loss: {:.4f}'.format(epoch+1, T, loss_ / i)) 
#测试孪生神经网络
net.eval() 
#读取数据集
test_set = enumerate(test_dataloader) 
ix,test_data = next(test_set) 
ims_1 = test_data[0] 
print(ims_1.size()) #查看第 1 组图像数据结构
ims_2 = test_data[1] 
print(ims_2.size()) #查看第 2 组图像数据结构 
label = test_data[2] 
print('Similarity Label:',label) #查看两组图像对应的相似标记
#显示指定两幅图像及其相似标记
ix=18 #指定图像序号
im_1 = ims_1[ix,:] #读取第 1 幅图像
show_image(im_1) 
im_2 = ims_2[ix,:] #读取第 2 幅图像
show_image(im_2) 
print('Similarity Label:',label[ix]) #查看两组图像对应的相似标记
#测试两幅图像之间的相似度
output1,output2 = net(ims_1, ims_2) 
siamese_distance = F.pairwise_distance(output1, output2) #距离
print('Siamese Distance:',siamese_distance) 
#通过设置阈值的方式求取精度
TH = 2 #距离阈值（距离小于阈值 TH 表明两幅图像相似）
siamese_distance[siamese_distance<2]=1 
siamese_distance[siamese_distance>=2]=0 
#求取精度
acc = torch.abs(siamese_distance - label).mean() 
#查看精度
print('Accuracy：{}'.format(acc))