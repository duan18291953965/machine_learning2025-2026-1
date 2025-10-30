# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:42:15 2024

@author: Administrator
"""

#导入 PyTorch 框架库
import torch 
from torch import nn, optim 
from torch.autograd import Variable 
import torch.nn.functional as F 
#导入视觉处理相关模块
from torch.utils.data import DataLoader 
from torchvision import datasets, transforms 
#每次处理的图像数量
batch_size = 64 
#预处理操作
data_transform = transforms.Compose( 
 [transforms.ToTensor(), 
 transforms.Normalize([0.5], [0.5])]) 
#加载数据
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, 
download=True) 
test_dataset = datasets.MNIST(root='./data', train=False, transform= data_transform) 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 
#定义卷积神经网络
class CNN(nn.Module): 
    def __init__(self): 
        super(CNN, self).__init__() 
 #输入通道为 1，输出通道为 20，卷积核尺寸为 5×5，步长为 1 
        self.conv1 = nn.Conv2d(1, 20, 5, 1) 
 #输入通道为 20，输出通道为 50，卷积核尺寸为 5×5，步长为 1 
        self.conv2 = nn.Conv2d(20, 50, 5, 1) 
        self.fc1 = nn.Linear(4*4*50, 500) #线性映射
        self.fc2 = nn.Linear(500, 10) #线性映射
    def forward(self, x): 
 # 输入图像尺寸为 1×28×28 
        x = F.relu(self.conv1(x)) #输出特征图像尺寸为 20×24×24 
        x = F.max_pool2d(x, 2, 2) #输出特征图像尺寸为 20×12×12 
        x = F.relu(self.conv2(x)) #输出特征图像尺寸为 50×8×8 
        x = F.max_pool2d(x, 2, 2) #输出特征图像尺寸为 50×4×4 
        x = x.view(-1, 4*4*50) #将特征图像拉伸为 4×4×50 维向量
        x = F.relu(self.fc1(x)) #将 4×4×50 维向量映射为 500 维向量
        x = self.fc2(x) #将 500 维向量映射为 10 维向量
        return x #返回结果
#实例化卷积神经网络
net = CNN() 
#定义多元交叉熵损失函数
Loss = nn.CrossEntropyLoss() 
#设置优化器
Opt = optim.SGD(net.parameters(), lr=0.02) 
#训练卷积神经网络
T = 10 #设置迭代次数
for epoch in range(T): 
    loss_ = 0.0 #累积训练误差
    acc_ = 0.0 #累积训练精度
    for i, data in enumerate(train_loader, 1): 
        im, label = data 
        label_pred = net(im) #前向传播
        L = Loss(label_pred, label) #计算误差
        loss_ += L.data.numpy() #误差累积
        _, label_opt = torch.max(label_pred, 1) #求取预测概率对应的类别
        acc_ += (label_opt == label).float().mean() #累积精度
        Opt.zero_grad() #梯度清零
        L.backward() #误差反传
        Opt.step() #更新参数
 #显示误差与精度变化
        if (epoch==0) | ((epoch+1) % 2 == 0): 
            print('Epoch:[{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch+1, T, loss_/ i, acc_ / i)) 
#测试卷积神经网络
net.eval() 
acc_ = 0.0 #累积精度
for i, data in enumerate(test_loader, 1): 
    im, label = data 
    label_pred = net(im) #前向传播
    _, label_opt = torch.max(label_pred, 1) 
    acc_+= (label_opt == label).float().mean() 
print('Accuracy: {:.4f}'.format(acc_ / i))