# -*- coding: utf-8 -*-
# @Author  : zhouxin
# @Time    : 2023/10/6 19:35
# @File    : Main.py.py
# @annotation    :
import math
from abc import ABC
import time
from numpy.linalg import *
from math import *
import gym
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
import random


# 链接Anylogic可视化展示
# 定义一个函数输出各点位置x,y坐标,
def position_print(swarm):
    with open("data_vsiual.txt", "w") as f:
        for agent in swarm:
            f.write("{},{},{},{},{},{}\n".format(agent.id, agent.pos[0], agent.pos[1], agent.vel[0]/3.5,
                                                 agent.vel[1]/3.5, agent.is_leader))


class Field:
    def __init__(self):
        self.width = 500  # x_max[m]
        self.height = 500  # y_max[m]


field = Field()


# def cal_angle_of_vector(v0, v1):
#     # print("{},{}".format(v0,v1))
#     v0 = v0/norm(v0)
#     v1 = v1/norm(v1)
#     dot_product = np.dot(v0, v1)
#     v0_len = np.linalg.norm(v0)
#     v1_len = np.linalg.norm(v1)
#     try:
#         angle_rad = np.arccos(dot_product / (v0_len * v1_len))
#     except ZeroDivisionError as error:
#         raise ZeroDivisionError("{}".format(error))
#     return angle_rad

# 计算两个方向的夹角
def cal_angle_of_vector(v1, v2):
    # print(v1,v2)
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    v1_norm = math.sqrt(v1[0] * v1[0] + v1[1] * v1[1])
    v2_norm = math.sqrt(v2[0] * v2[0] + v2[1] * v2[1])
    cons = dot_product / (v1_norm * v2_norm)
    if cons<-1:
        cons = -1
    if cons > 1:
        cons = 1
    angle_rad = math.acos(cons)
    return angle_rad

# 随机选择领导者
def get_n_rand(n, p):
    leader_list = set()
    while True:
        leader_list.add(random.randint(0, n - 1))
        if len(leader_list) == n * p:
            break
    return leader_list


# 角度旋转
def rotation_matrix_about(v, angle):
    x = v[1] * math.sin(angle) + v[0] * math.cos(angle)
    y = v[1] * math.cos(angle) + v[0] * math.sin(angle)
    return [x, y]


# 定义一个智能体的类
class Agent:
    def __init__(self, agent_id, speed):
        self.id = agent_id
        # 位置
        self.pos = np.array([0, 0])
        self.pos[0] = np.random.uniform(100, 100)
        self.pos[1] = np.random.uniform(100, 100)
        # 速度
        self.vel = np.random.uniform(-5, 5, 2)

        # 各个方向的速度分量
        self.vel = self.vel / norm(self.vel) * speed

        # 目标影响权重
        self.w_p = 0.5

        # 目标方向
        self.g = np.array([1, 1]) / norm(np.array([1, 1]))

        # 是否被选为领导
        self.is_leader = False

    def update_position(self, delta_t):
        self.pos = self.pos + self.vel * delta_t


class Couzin(gym.Env):
    # 初始化
    """
    1. 初始化集群数量
    2. 初始化位置
    3. 初始化速度方向
    4. 初始化速度
    5. 初始化排斥距离
    6. 初始化吸引距离
    7. 初始化角速度
    """

    def __init__(self, N):
        # 初始化参数
        # 初始化集群中个体数量
        self.n = N
        # 初始化排斥距离
        self.a_minimal_range = 2
        # 初始化吸引距离
        self.attract_range = 5
        # 初始化速度
        self.constant_speed = 1
        # 初始化角速度
        self.theta_dot_max = 1
        # 初始化领导者比例
        self.p = 0.3
        # swarm 生成集群
        self.swarm = []
        [self.swarm.append(Agent(i, self.constant_speed)) for i in range(self.n)]

        # 设置时间步长
        self.dt = 0.1

        ##################################################
        # field_of_view 可修改
        self.field_of_view = 3 * pi / 2



        # 生成领导者
        self.leader_list = get_n_rand(self.n, self.p)

        # 更新领导者标志
        for i in range(len(self.swarm)):
            for leader_id in self.leader_list:
                if i == leader_id:
                    self.swarm[i].is_leader = True
        self.fig = plt.figure()
        self.ax = self.fig.gca()

    # 核心函数
    # 奖励函数-运动趋势
    # 分裂-惩罚 平均空间相关度
    # 整体reward 到达目标点大的reward
    def step(self):
        # actions 是一个集合，包含追随者的可视角和领综合

        # 遍历集群
        for agent in self.swarm:
            # 2005 couzin领导模型
            d = 0
            dr = 0
            da = 0
            dv = agent.vel

            for neighbor in self.swarm:
                if agent.id != neighbor.id:
                    # 位置向量，单位位置向量，距离
                    r = neighbor.pos - agent.pos
                    r_normalized = r / norm(r)
                    norm_r = norm(r)
                    # 速度向量
                    agent_vel_normalized = agent.vel / norm(agent.vel)
                    if cal_angle_of_vector(r_normalized, agent_vel_normalized) < self.field_of_view / 2:
                        if norm_r < self.a_minimal_range:
                            dr = dr - r_normalized
                        elif norm_r < self.attract_range:
                            da = da + r_normalized
                            dv = dv + neighbor.vel / norm(neighbor.vel)
            if norm(dr) != 0:
                if agent.is_leader:
                    dr = dr / norm(dr)
                    d = (dr + agent.w_p * agent.g) / norm(dr + agent.w_p * agent.g)
                else:
                    d = dr / norm(dr)
            elif norm(da) != 0:
                if agent.is_leader:
                    d_new = (da + dv) / norm(da + dv)
                    d = (d_new + agent.w_p * agent.g) / norm(d_new + agent.w_p * agent.g)
                else:
                    d_new = (da + dv) / norm(da + dv)
                    d = d_new

            if norm(d) != 0:
                angle_between = cal_angle_of_vector(d, agent.vel)
                if angle_between >= self.theta_dot_max * self.dt:
                    rot = rotation_matrix_about(d, self.theta_dot_max * self.dt)

                    vel0 = rot

                    rot1 = rotation_matrix_about(d, -self.theta_dot_max * self.dt)

                    vel1 = rot1

                    if cal_angle_of_vector(vel0, d) < cal_angle_of_vector(vel1, d):
                        agent.vel = vel0 / norm(vel0) * self.constant_speed
                    else:
                        agent.vel = vel1 / norm(vel1) * self.constant_speed
                else:
                    agent.vel = d / norm(d) * self.constant_speed
            # 更新各个点的坐标位置
            [agent.update_position(self.dt) for agent in self.swarm]
            # 输出各个智能体的编号，坐标，速度方向,是否是领导者
            position_print(self.swarm)


            # 可视化展示

            x = np.array([])
            y = np.array([])
            z = np.array([])
            x_dot = np.array([])
            y_dot = np.array([])
            z_dot = np.array([])
            for agent in self.swarm:
                # x，y,z 分别存储x,y
                x = np.append(x, agent.pos[0])
                y = np.append(y, agent.pos[1])

                # x_dot，y_dot 分别存储x,y
                x_dot = np.append(x_dot, agent.vel[0])
                y_dot = np.append(y_dot, agent.vel[1])


            self.ax.clear()


            # 设置箭头的形状和大小
            x_temp = np.array([])
            y_temp = np.array([])


            x_temp_dot = np.array([])
            y_temp_dot = np.array([])


            for i in range(len(self.swarm)):
                for leader_id in list(self.leader_list):
                    if self.swarm[i].id == leader_id:
                        continue
                    # x，y分别存储x,y方向上的位置
                    x_temp = np.append(x_temp, self.swarm[i].pos[0])
                    y_temp = np.append(y_temp, self.swarm[i].pos[1])

                    # x_dot，y_dot 分别存储x,y方向上的范数
                    x_temp_dot = np.append(x_temp_dot, self.swarm[i].vel[0])
                    y_temp_dot = np.append(y_temp_dot, self.swarm[i].vel[1])

            self.ax.quiver(x_temp, x_temp, x_temp_dot, y_temp_dot, width=0.02,
                      scale=10,units="inches", color='#EC3684')

            for leader_id in list(self.leader_list):
                self.ax.quiver(x[leader_id], y[leader_id], self.swarm[leader_id].vel[0],
                          self.swarm[leader_id].vel[1],
                          width=0.02, scale=10, units="inches", color='#006400')

            self.ax.set_aspect('auto', 'box')
            self.ax.set_xlim(0, field.width)
            self.ax.set_ylim(0, field.height)

            self.ax.tick_params(axis='x', colors='red')
            self.ax.tick_params(axis='y', colors='blue')

        plt.pause(0.1)


couzin = Couzin(10)
for i in range(100):
    couzin.step()
