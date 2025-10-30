# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:19:46 2024

@author: Administrator
"""

#导入神经网络模块
import torch 
from torch import nn 
#from torch.autograd import Variable 
#导入绘图库
import matplotlib.pyplot as plt 
#导入科学计算库
import numpy as np 
#导入 make_blobs 数据集
from sklearn.datasets import make_blobs 
#构造数据
X, y = make_blobs(n_samples=500, n_features=2, centers=[[0,0], [1,2]], cluster_std=[0.4, 0.5]) 
#转换为类型
x_train = torch.tensor(X).type(torch.FloatTensor) 
y_train = torch.tensor(y).type(torch.FloatTensor) 
#构建神经网络
#自定义神经网络类
class L_NN(nn.Module): 
    def __init__(self,n_feature): 
        super(L_NN, self).__init__() 
        self.lr = nn.Linear(n_feature, 1) 
        self.predict = nn.Sigmoid() 
    def forward(self, x): 
        x_=self.lr(x) 
        y = self.predict(x_) 
        return y.squeeze(-1) #调整格式使其与训练数据中的 y 值格式相一致
#实例化神经网络对象
net = L_NN(2) #两个维度对应两个特征
# 定义损失函数
Loss = nn.BCELoss() #二元交叉熵损失函数
#定义优化器
Opt = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9) 
#训练神经网络
T=1000 #设定迭代次数
for epoch in range(T): 
    y_pred = net(x_train) #前向传播
    L = Loss(y_pred,y_train) #计算损失
    Opt.zero_grad() #梯度清零
    L.backward() #误差反传
    Opt.step() #更新参数
 #计算预测精度
    label = y_pred.ge(0.5).float() #以 0.5 为阈值进行分类
    acc = (label == y_train).float().mean() #计算精度
 #每 10 轮显示一次误差与精度
    if (epoch==0) | ((epoch+1) % 10 == 0): 
        print('Epoch:[{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch+1, T, L.item(), acc)) 
#分类结果可视化
#获取模型参数
w0, w1 = net.lr.weight[0] 
w0 = float(w0.item()) 
w1 = float(w1.item()) 
b = float(net.lr.bias.item()) 
plot_x = np.arange(min(X[:,0]), max(X[:,0]), 0.1) 
plot_y = (-w0 * plot_x - b) / w1 
plt.figure() 
plt.scatter(X[:, 0], X[:, 1], marker='o',s=50,c=y, cmap='RdYlGn',linewidths=1, 
edgecolors='k') 
plt.plot(plot_x, plot_y,color='b',lw=4) 
plt.show() 
#查看参数
para_list = list(net.parameters()) 
print(para_list) 
