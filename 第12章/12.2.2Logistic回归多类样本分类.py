# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:23:48 2024

@author: Administrator
"""
#导入绘图库
import matplotlib.pyplot as plt 
#导入 PyTorch 框架库
import torch 
from torch import nn 
from torch.autograd import Variable 
import numpy as np #导入科学计算模块
#导入视觉处理相关模块
import torchvision 
from torchvision import datasets 
from torchvision import transforms 
#设置参数
batch_size = 32 #每次处理的图像数量
N = 28 #图像尺寸
#图像预处理操作（只进行张量化与归一化）
img_transform = transforms.Compose([ 
 transforms.ToTensor(), 
 transforms.Normalize(mean=(0.5, ), std=(0.5, )) #灰度图的均值与方差均为 0.5 
]) 
#加载 MNIST 数据集
train_dataset = datasets.MNIST(root='./data/', train=True, transform=img_transform, 
download=True) 
test_dataset = datasets.MNIST(root='./data/', train=False, transform=img_transform, 
download=True) 
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, 
shuffle=True) 
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, 
shuffle=True) 
#构建神经网络
class L_NN2(nn.Module): 
    def __init__(self, in_dim, n_class): 
        super(L_NN2, self).__init__() 
        self.logstic = nn.Linear(in_dim, n_class) #直接进行线性映射
    def forward(self, x): 
        y = self.logstic(x) 
        return y 
net = L_NN2(N*N, 10) #图像分辨率为 28 像素×28 像素
#神经网络训练
Loss = nn.CrossEntropyLoss() #多元交叉熵损失函数
Opt = torch.optim.SGD(net.parameters(), lr=1e-3) #定义优化器
T=100 #设置迭代次数
for epoch in range(T): 
    loss_ = 0.0 #累积本轮训练误差
    acc_ = 0.0 #累积本轮训练精度
    for i, data in enumerate(train_loader, 1): 
        im, label = data #读取图像与类别标记
        im = im.view(im.size(0), -1) #将图像拉伸为 28×28 维向量
        label_pred = net(im) #前向传播
        L = Loss(label_pred, label) #计算误差
        loss_ += L.data.numpy() #误差累积
        _, label_opt = torch.max(label_pred, 1) #求取预测概率最大者对应的类别
        acc_ += (label_opt == label).float().mean() #累积精度
        Opt.zero_grad() #梯度清零
        L.backward() #误差反传
        Opt.step() #更新参数
#输出每轮代价值与精度
        if (epoch==0) | ((epoch+1) % 10 == 0): 
            print('Epoch:[{}/{}], Loss:{:.4f}, Accuracy:{:.4f}'.format(epoch+1, T, loss_/i, acc_/i)) 
#测试神经网络
net.eval() 
acc_ = 0.0 #累积精度
for i, data in enumerate(test_loader, 1): 
    im, label = data #读取图像与类别标记
    im = im.view(im.size(0), -1) #将图像拉伸为 28×28 维向量
    label_pred = net(im) #前向传播
    _, label_opt = torch.max(label_pred, 1) #求取预测概率最大者对应的类别
    acc_ += (label_opt == label).float().mean() #累积精度
print('Accuracy: {:.4f}'.format(acc_/i)) 
#指定图像进行测试
test_set = enumerate(test_loader) 
idx,(test_data,test_labels) = next(test_set) #读取 batch_size 指定数量的图像
#提取图像与类别标记
im_test = test_data[15,:] #读取序号为 15 的图像
im_test = im_test.view(im_test.size(0), -1) #将图像拉伸为 28×28 维向量
y_pred = net(im_test) #利用已训练的神经网络预测类别
_, y_pred = torch.max(y_pred.data, 1) #求取预测概率最大者对应的类别
#显示图像与预测的类别标记
im = im_test.reshape(28,28) #将 28×28 维向量还原为图像尺寸
#显示结果
plt.figure() 
plt.imshow(im,cmap='gray') 
plt.title('Predicted result:'+str(y_pred.item()))