# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:32:18 2024

@author: Administrator
"""

#导入 PyTorch 框架库
import torch 
import torch.nn as nn 
from torch.autograd import Variable 
#导入视觉处理相关库
from torch.utils.data import DataLoader 
from torchvision import datasets, transforms 
#导入绘图库
import matplotlib.pyplot as plt 
batch_size=32 
#加载数据（仅做张量化预处理）
train_dataset=datasets.MNIST(root='./data/',train=True,transform=transforms.ToTensor(), 
download=False) 
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor()) 
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) 
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False) 
#构建自动编码解码器
#定义自动编码解码器类
class AutoEncoder(nn.Module): 
    def __init__(self): 
        super(AutoEncoder, self).__init__() 
        #编码（层次、每层维度与激活函数类型可调整）
        self.encoder = nn.Sequential( 
            nn.Linear(28*28, 128), 
            nn.Tanh(), 
            nn.Linear(128, 64), 
            nn.Tanh(), 
            nn.Linear(64, 12), 
            nn.Tanh(), 
            nn.Linear(12, 3), 
 ) 
 #解码（层次、每层维度与激活函数类型可调整；注意与编码过程的对应）
        self.decoder = nn.Sequential( 
            nn.Linear(3, 12), 
            nn.Tanh(), 
            nn.Linear(12, 64), 
            nn.Tanh(), 
            nn.Linear(64, 128), 
            nn.Tanh(), 
            nn.Linear(128, 28*28), 
            nn.Sigmoid(), 
 ) 
    def forward(self, x): 
        encoded = self.encoder(x) #数据编码
        decoded = self.decoder(encoded) #数据解码
        return encoded, decoded #返回结果
#实例化自动编码解码器对象 
autoencoder = AutoEncoder() 
#训练自动编码解码器
#定义损失函数
Loss = nn.MSELoss() #标准差损失函数
#设置优化器
Opt = torch.optim.Adam(autoencoder.parameters(), lr=0.005) 
#开始训练
T = 50 #设定训练次数
for epoch in range(T): 
    loss_ = 0.0 #累积训练误差
    for i, (x, y) in enumerate(train_loader, 1): 
        b_x = Variable(x.view(-1, 28*28)) #将图像拉伸为 28×28 维向量
        b_y = Variable(x.view(-1, 28*28)) #与 b_x 相同
        encoded, decoded = autoencoder(b_x) #编码与解码
        L = Loss(decoded, b_y) #计算误差
        loss_ += L.item() #误差累积
        Opt.zero_grad() #梯度清零
        L.backward() #误差反传
        Opt.step() #更新参数
 #显示误差变化
    if (epoch==0) | ((epoch+1) % 10 == 0): 
        print('Epoch:[{}/{}], Loss: {:.4f}'.format(epoch+1, T, loss_/i)) 
#测试自动编码解码器
x = test_dataset.data[1:2] #读取图像
im = x.numpy().reshape(28,28) #将 28×28 维向量还原为图像尺寸
#显示真实图像
plt.figure(1) 
plt.imshow(im,cmap='gray') 
#利用自动编码解码器提取特征
#编码
x_ = Variable(x.view(-1, 28*28).type(torch.FloatTensor)/255.) 
encoded_data, _ = autoencoder(x_) 
#解码
decoded_data = autoencoder.decoder(encoded_data) 
#结果显示
plt.figure(2) 
decoded_im = decoded_data.detach().numpy().reshape(28,28) #将 28×28 维向量还原为图像尺寸
plt.imshow(decoded_im,cmap='gray')
#随机数生成图像
code = torch.FloatTensor([[0.1, -0.9, 0.3]]) 
decode = autoencoder.decoder(code) 
decode = decode.view(decode.size()[0], 28, 28) 
decoded_im = decode.detach().numpy().reshape(28,28) 
plt.imshow(decoded_im, cmap='gray')