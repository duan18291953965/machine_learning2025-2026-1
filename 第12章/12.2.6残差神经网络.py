# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:07:33 2024

@author: Administrator
"""

#导入 PyTorch 框架库及 Torchvision 库
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision import datasets 
from torchvision import transforms 
from torch.utils.data import DataLoader 
#每次处理的图像数量
batch_size = 128 
#预处理操作
data_transform = transforms.Compose( 
 [transforms.ToTensor(), #转换为张量
 transforms.Normalize([0.5], [0.5])]) #归一化
#加载数据
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, 
download=False) 
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform) 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 
#定义残差神经网络模块
class ResidualBlock(nn.Module): 
    def __init__(self, channels): 
        super(ResidualBlock, self).__init__() 
        self.channels = channels 
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1) 
    def forward(self, x): 
        x = self.conv1(x) 
        y = self.conv2(F.relu(x)) 
        y += x 
        y = F.relu(y) 
        return y 
#定义残差神经网络
class NET(nn.Module): 
    def __init__(self): 
        super(NET, self).__init__() 
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5) 
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.mp = nn.MaxPool2d(2) 
        self.rblock1 = ResidualBlock(16) #导入残差神经网络模块（通道数为 16）
        self.rblock2 = ResidualBlock(32) #导入残差神经网络模块（通道数为 32）
        self.fc = nn.Linear(512, 10) 
    def forward(self, x): 
        x = self.conv1(x) 
        x = self.mp(F.relu(x)) 
        x = self.rblock1(x) 
        x = self.conv2(x) 
        x = self.mp(F.relu(x)) 
        x = self.rblock2(x) 
        x = x.view(x.size(0), -1) 
        x = self.fc(x) 
        return x 
#实例化神经网络对象
model = NET() 
#定义损失函数（多元交叉熵损失函数）
loss = torch.nn.CrossEntropyLoss() 
#定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=.5) 
#训练神经网络
T = 10 #训练迭代次数
for epoch in range(T): 
    #running_loss = 0 
    loss_ = 0.0 #累积训练误差
    acc_ = 0.0 #累积训练精度
    for i, data in enumerate(train_loader):
        im, label = data 
        label_pred = model(im) #前向传播
        L = loss(label_pred, label) #计算误差
        loss_ += L.data.numpy() #误差累积
        _, label_opt = torch.max(label_pred, 1) #求取预测概率对应的类别
        acc_ += (label_opt == label).float().mean() #累积精度
        optimizer.zero_grad() #梯度清零
        L.backward() #误差反传
        optimizer.step() #更新参数
 #显示误差与精度变化
    if (epoch==0) | ((epoch+1) % 2 == 0): 
        print('Epoch:[{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch+1, T, loss_ / i, acc_ / i)) 
#测试神经网络
model.eval() 
acc_ = 0.0 #累积精度
for i, data in enumerate(test_loader, 1): 
    im, label = data 
    label_pred = model(im) #前向传播
    _, label_opt = torch.max(label_pred, 1) 
    acc_+= (label_opt == label).float().mean() 
print('Accuracy: {:.4f}'.format(acc_ / i))