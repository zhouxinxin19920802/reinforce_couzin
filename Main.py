# -*- coding: utf-8 -*-
# @Author  : zhouxin
# @Time    : 2023/10/6 19:35
# @File    : Main.py.py
# @annotation    :
import math

from numpy.linalg import *
from math import *
import gym
from gym import spaces

import numpy as np
import random


class Field:
    def __init__(self):
        self.width = 1000  # x_max[m]
        self.height = 1000  # y_max[m]


field = Field()


def cal_angle_of_vector(v0, v1):
    dot_product = np.dot(v0, v1)
    v0_len = np.linalg.norm(v0)
    v1_len = np.linalg.norm(v1)
    try:
        angle_rad = np.arccos(dot_product / (v0_len * v1_len))
    except ZeroDivisionError as error:
        raise ZeroDivisionError("{}".format(error))
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
        self.pos[0] = np.random.uniform(field.width * 1 / 3, field.width * 2 / 3)
        self.pos[1] = np.random.uniform(field.height * 1 / 3, field.height * 2 / 3)
        self.vel = np.random.uniform(-5, 5, 2)



        # 各个方向的速度分量
        self.vel = self.vel / norm(self.vel) * speed

        self.w_p = 0.5

        # 目标方向
        self.g = np.array([-4, 5]) / norm(np.array([-4, 5]))

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

    def _init__(self, N):
        # 初始化参数
        # 初始化集群中个体数量
        self.n = N
        # 初始化排斥距离
        self.a_minimal_range = 2
        # 初始化吸引距离
        self.attract_range = 5
        # 初始化速度
        self.constant_speed = 3.5
        # 初始化角速度
        self.theta_dot_max = 1
        # 初始化领导者比例
        self.p = 0.1
        # swarm 生成集群
        self.swarm = []
        [self.swarm.append(Agent(i, self.constant_speed)) for i in range(self.n)]

        # 设置时间步长
        self.dt = 0.1

        ##################################################
        # field_of_view 可修改
        self.field_of_view = 3 * pi / 2

        # 领导者影响权重比

        # 生成领导者
        self.leader_list = get_n_rand(self.n, self.p)

        # 更新领导者标志
        for i in range(len(self.swarm)):
            for leader_id in self.leader_list:
                if i == leader_id:
                    self.swarm[i].is_leader = True

    # 核心函数
    # 奖励函数-运动趋势
    # 分裂-惩罚 平均空间相关度
    # 整体reward 到达目标点大的reward
    def step(self, actions):
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
                    if acos(np.dot(r_normalized, agent_vel_normalized)) < self.field_of_view / 2:
                        if norm_r < self.a_minimal_range:
                            dr = dr - r_normalized
                        elif norm_r < self.attract_range:
                            da = da + r_normalized
                            dv = dv + neighbor.vel / norm(neighbor.vel)
            if norm(dr) != 0:
                if agent.is_leader == True:
                    dr = dr / norm(dr)
                    d = (dr + Agent.w_p * Agent.g) / norm(dr + Agent.w_p * Agent.g)
                else:
                    d = dr / norm(dr)
            elif norm(da) != 0:
                if agent.is_leader == True:
                    d_new = (da + dv) / norm(da + dv)
                    d = (d_new + Agent.w_p * Agent.g) / norm(d_new + Agent.w_p * Agent.g)
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

        observation = []
        reward = 5
        done = False

        return observation, reward, done
