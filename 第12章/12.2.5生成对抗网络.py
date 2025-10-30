# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:00:19 2024

@author: Administrator
"""

#导入 PyTorch 框架库及 Torchvision 库
import torch 
#import torchvision 
import torch.nn as nn 
#import torch.nn.functional as F 
from torchvision import datasets 
from torchvision import transforms 
from torchvision.utils import save_image 
from torch.autograd import Variable 
from torch.utils.data import DataLoader 
#图像显示函数
def to_im(x): 
    im = 0.5 * (x + 1) 
    im = im.clamp(0, 1)
    im = im.view(-1, 1, 28, 28) 
    return im 
#每次处理的图像数量
batch_size = 64 
#生成图像的噪声向量维度
Z = 200 
#构造数据
img_transform = transforms.Compose([ 
 transforms.ToTensor(), 
 transforms.Normalize(mean=(0.5, ), std=(0.5, )) 
]) 
#预处理操作
data_transform = transforms.Compose( 
 [transforms.ToTensor(), 
 transforms.Normalize([0.5], [0.5])]) 
#加载数据
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, 
download=True) 
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform) 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 
#定义生成对抗网络
class discriminator(nn.Module): 
    def __init__(self): 
        super(discriminator, self).__init__() 
        self.discriminator = nn.Sequential( 
            nn.Linear(784, 256),
            nn.ReLU(True), 
            nn.Linear(256, 128), 
            nn.ReLU(True), 
            nn.Linear(128, 1), 
            nn.Sigmoid() 
 ) 
    def forward(self, x): 
        x = self.discriminator(x) 
        return x 
class generator(nn.Module): 
    def __init__(self,input_size): 
        super(generator, self).__init__() 
        self.generator = nn.Sequential( 
            nn.Linear(input_size, 128), 
            nn.ReLU(True), 
            nn.Linear(128, 256), 
            nn.ReLU(True), 
            nn.Linear(256, 784), 
            nn.Tanh() 
 ) 
    def forward(self, x): 
        x = self.generator(x) 
        return x 
#训练生成对抗网络
D = discriminator() #实例化判别器
G = generator(Z) #实例化生成器
loss = nn.BCELoss() #定义二元交叉熵损失函数
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001) #定义判别器的优化器
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001) #定义生成器的优化器
T = 100 #训练迭代次数
for epoch in range(T): 
    for i, (im, _) in enumerate(train_loader): 
        num_im = im.size(0) 
        #训练判别器
        im = im.view(num_im, -1) 
        real_im = Variable(im) 
        real_label = Variable(torch.ones(num_im)) 
        fake_label = Variable(torch.zeros(num_im)) 
        real_pred = D(real_im).squeeze(-1) #预测真图像的类别标记（理想情况为 1）
        d_loss_real = loss(real_pred, real_label) #真图像对应损失（预测类别标记、真实类别标记）
        # real_scores = real_pred 
        z_vector = Variable(torch.randn(num_im, Z)) #生成噪声向量
        fake_img = G(z_vector) #生成假图像
        fake_pred = D(fake_img).squeeze(-1) #判别器对假图像的预测类别标记（理想情况为 0）
        d_loss_fake = loss(fake_pred, fake_label) #假图像对应损失（预测类别标记、真实类别标记）
        # fake_scores = fake_pred 
        d_loss = d_loss_real + d_loss_fake #真假图像损失之和
        d_optimizer.zero_grad() #梯度清零
        d_loss.backward() #误差反传
        d_optimizer.step() #更新参数
        #训练生成器
        z_vector = Variable(torch.randn(num_im, Z)) #生成噪声向量
        fake_im = G(z_vector) #生成假图像
        fake_pred = D(fake_im).squeeze(-1) #判别器对假图像的预测类别标记
        g_loss = loss(fake_pred, real_label) #计算损失（所生成的图像尽可能逼近真实图像）
        g_optimizer.zero_grad() #梯度清零
        g_loss.backward() #误差反传
        g_optimizer.step() #更新参数
        if (i + 1) % 100 == 0: 
            print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '.format(epoch, T, d_loss.data.numpy(), g_loss.data.numpy(),)) 
    if epoch == 0: 
        real_images = to_im(real_im.data)
        save_image(real_images, './results/real_images.png') 
        fake_images = to_im(fake_img.data) 
        save_image(fake_images, './results/fake_images-{}.png'.format(epoch + 1))