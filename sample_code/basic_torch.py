# -*- coding: utf-8 -*-
# @Author  : zhouxin
# @Time    : 2023/10/27 16:41
# @File    : basic_torch.py
# @annotation    :
import logging

logging.basicConfig(
    level=logging.DEBUG,  # 控制台打印的日志级别
    filename="test_log.txt",
    filemode="w",  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
    # a是追加模式，默认如果不写的话，就是追加模式
    format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    # 日志格式
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(3) + 0.1 * torch.randn(x.size())
logging.info("x:{}".format(x))
# logging.info("x_size:{}".format(x.size()))
# logging.info("x_value:{}".format(torch.randn(x.size())))
# logging.info("y:{}".format(y))

# x , y =(Variable(x),Variable(y))

# plt.scatter(x.detach(),y.detach())
# # 或者采用如下的方式也可以输出x,y
# # plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()


class Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(n_input, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, input):
        out = self.hidden1(input)
        out = F.relu(out)
        out = self.hidden2(out)
        out = F.sigmoid(out)
        out = self.predict(out)

        return out


net = Net(1, 20, 1)
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
loss_func = torch.nn.MSELoss()
for t in range(5000):
    prediction = net(x)
    loss = loss_func(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # if t % 5 == 0:
    #     plt.cla()
    #     plt.scatter(x.data.numpy(), y.data.numpy())
    #     plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    #     plt.text(0.5, 0, 'Loss = %.4f' % loss.data, fontdict={'size': 20, 'color': 'red'})
    #     plt.pause(0.05)

# 保存训练好的整个网络
torch.save(net, "net1.pth")
# 只保存训练好的网络的参数
torch.save(net.state_dict(), "net1_params.pth")

net2 = torch.load("net1.pth")
x = torch.tensor([5], dtype= torch.float)

y = net2(x)
print("y:{}".format(y))
