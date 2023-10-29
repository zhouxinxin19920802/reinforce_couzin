# -*- coding: utf-8 -*-
# @Author  : zhouxin
# @Time    : 2023/10/29 9:09
# @File    : gradient_rise.py
# @annotation    :

import torch
import torch.optim as optim


def f(x):
    return - x ** 2 + 10


x = torch.tensor([5.0], requires_grad=True)

optimizer = optim.SGD([x], lr=0.1)



for _ in range(100):
    optimizer.zero_grad()
    loss = - f(x)
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新 x

# 打印最终的 x 值，它应该接近函数的最大值
print("最大值 x:", x.item())
print("最大值 f(x):", f(x).item())