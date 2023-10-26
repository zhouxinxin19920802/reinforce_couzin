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

# 日志模块
import logging

logging.basicConfig(
    level=logging.DEBUG,  # 控制台打印的日志级别
    filename="test_log.txt",
    filemode="w",  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
    # a是追加模式，默认如果不写的话，就是追加模式
    format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    # 日志格式
)


# 链接Anylogic可视化展示
# 定义一个函数输出各点位置x,y坐标,
def position_print(swarm):
    for agent in swarm:
        logging.info(
            "{},{},{},{},{},{}\n".format(
                agent.id, agent.pos[0], agent.pos[1], agent.vel[0] / 3.5, agent.vel[1] / 3.5, agent.is_leader
            )
        )


class Field:
    def __init__(self):
        self.width = 500
        self.height = 500


field = Field()


# 计算两个方向的夹角
def cal_angle_of_vector(v1, v2):
    # print(v1,v2)
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    # logging.info("v1_norm_v2_norm:{},{},{}".format(v1,v2,norm(v1) * norm(v2) ))
    cons = dot_product / (norm(v1) * norm(v2))
    if cons < -1:
        cons = -1
    if cons > 1:
        cons = 1
    angle_rad = math.acos(cons)
    return angle_rad


# 随机选择领导者
# 这个只是在集群中选定位置
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


# 定义工具函数，计算两个智能体之间的距离
def cal_distance(agent_a, agent_b):
    distance_vector = agent_a.pos - agent_b.pos
    return norm(distance_vector)


# 定义一个智能体的类
# 包含数据和更新位置
class Agent:
    def __init__(self, agent_id, speed):
        self.id = agent_id
        # 位置
        self.pos = np.array([0, 0])
        self.pos[0] = np.random.uniform(70, 100)
        self.pos[1] = np.random.uniform(70, 100)
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

        # 每个点的邻域集合
        self.neibour_set = []

    def update_position(self, delta_t):
        self.pos = self.pos + self.vel * delta_t
        if self.pos[0] < 0:
            self.pos[0] = self.pos[0] + field.width
        if self.pos[0] > field.width:
            self.pos[0] = self.pos[0] - field.width
        if self.pos[1] > field.height:
            self.pos[1] = self.pos[1] - field.height
        if self.pos[1] < 0:
            self.pos[1] = self.pos[1] + field.height


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

    def __init__(self):
        # 初始化参数
        # 初始化集群中个体数量

        self.n = 20
        # 初始化排斥距离
        self.a_minimal_range = 10
        # 初始化吸引距离
        self.attract_range = 50
        # 初始化速度
        self.constant_speed = 7
        # 初始化角速度
        self.theta_dot_max = 1
        # 初始化领导者比例
        self.p = 0.3
        # swarm 生成集群
        self.swarm = []
        [self.swarm.append(Agent(i, self.constant_speed)) for i in range(self.n)]
        # 需要修改，个体位置初始化时候
        # 生成第一个个体的坐标
        self.observation = []
        self.reward = 0
        self.steps = 0

        # 设置时间步长
        self.dt = 0.2

        ##################################################
        # field_of_view 可修改
        self.field_of_view = 4 * pi / 2

        # 生成领导者
        self.leader_list = get_n_rand(self.n, self.p)
        logging.info("leader_num:{}".format(self.leader_list))

        # 更新领导者标志
        for j in range(len(self.swarm)):
            for leader_id in self.leader_list:
                if j == leader_id:
                    self.swarm[j].is_leader = True
        self.fig = plt.figure()
        self.ax = self.fig.gca()

        #  定义目标点位置
        self.target_x = 450
        self.target_y = 450
        self.target_radius = 10

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
            # 排斥域
            dr = 0
            # 吸引域
            da = 0
            # 当前个体的速度
            dv = agent.vel

            if agent.is_leader:
                agent.g = np.array([self.target_x, self.target_y]) - agent.pos
                agent.g = agent.g / norm(agent.g)

            neighbor_count = 0

            # 这边要做个判断，如果已经到达目标范围内，则直接更新速度方向
            if math.sqrt(
                    pow(agent.pos[0] - self.target_x, 2) + pow(agent.pos[1] - self.target_y, 2)) < self.target_radius:
                agent.vel = np.array([self.target_x - agent.pos[0], self.target_y - agent.pos[1]])
                agent.vel = agent.vel / norm(agent.vel)
            else:
                for neighbor in self.swarm:
                    visual_vector = np.array([neighbor.pos[0] - agent.pos[0], neighbor.pos[1] - agent.pos[1]])
                    if agent.id != neighbor.id and cal_distance(agent,
                                                                neighbor) < self.attract_range and cal_angle_of_vector(
                        visual_vector, agent.vel) < self.field_of_view / 2:
                        agent.neibour_set.append(neighbor)
                        neighbor_count = neighbor_count + 1
                        # 位置向量，单位位置向量，距离
                        r = neighbor.pos - agent.pos
                        # logging.info("r:{}".format(r))
                        if norm(r) == 0:
                            r_normalized = r
                        else:
                            r_normalized = r / norm(r)
                        # 位置向量标准化
                        norm_r = norm(r)
                        # 速度向量
                        agent_vel_normalized = agent.vel / norm(agent.vel)
                        if cal_angle_of_vector(r_normalized, agent_vel_normalized) < self.field_of_view / 2:
                            if norm_r < self.a_minimal_range:
                                # 排斥区域，位置累计
                                dr = dr - r_normalized
                            elif norm_r < self.attract_range:
                                # 吸引区域位置向量累计
                                da = da + r_normalized
                                # 吸引区速度向量累计
                                dv = dv + neighbor.vel / norm(neighbor.vel) + agent.vel
                if norm(dr) != 0:
                    # 排斥区域
                    # if agent.is_leader:
                    #     dr = dr / norm(dr)
                    #     d = (dr + agent.w_p * agent.g) / norm(dr + agent.w_p * agent.g)
                    # else:
                    #     d = dr / norm(dr)
                    d = dr / norm(dr)
                elif norm(da) != 0:
                    # 吸引区域
                    if agent.is_leader:
                        d_new = (da + dv) / norm(da + dv)
                        d = (d_new + agent.w_p * agent.g) / norm(d_new + agent.w_p * agent.g)
                    else:
                        d_new = (da + dv) / norm(da + dv)
                        d = d_new
                else:
                    if agent.is_leader:
                        d = agent.g

                if norm(d) != 0:
                    angle_between = cal_angle_of_vector(d, agent.vel)
                    # logging.info("angle_between:{}".format(angle_between))
                    if angle_between >= self.theta_dot_max * self.dt:
                        # rotation_matrix_about 旋转后，返回的是向量
                        rot = rotation_matrix_about(d, self.theta_dot_max * self.dt)

                        vel0 = rot

                        rot1 = rotation_matrix_about(d, -self.theta_dot_max * self.dt)

                        vel1 = rot1

                        if cal_angle_of_vector(vel0, d) < cal_angle_of_vector(vel1, d):
                            agent.vel = vel0 / norm(vel0) * self.constant_speed
                        else:
                            agent.vel = vel1 / norm(vel1) * self.constant_speed
                    else:
                        agent.vel = d * self.constant_speed

        # 更新各个点的坐标位置
        [agent.update_position(self.dt) for agent in self.swarm]
        # 输出各个智能体的编号，坐标，速度方向,是否是领导者
        # logging.info("#########################")
        # for i in range(len(self.swarm)):
        #     logging.info("swarm:{},{},{}".format(self.swarm[i].id, self.swarm[i].pos, self.swarm[i].vel))
        self.steps = self.steps + 1
        if self.steps > 1000:
            done = True

        observation = 0
        reward = 0

        # return observation, reward,done

        # 可视化展示

        x = np.array([])
        y = np.array([])

        x_dot = np.array([])
        y_dot = np.array([])

        for agent in self.swarm:
            # 存储所有横纵坐标位置
            # x，y分别存储x,y
            x = np.append(x, agent.pos[0])
            y = np.append(y, agent.pos[1])

            # 存储所有横纵速度方向
            # x_dot，y_dot 分别存储x,y
            x_dot = np.append(x_dot, agent.vel[0])
            y_dot = np.append(y_dot, agent.vel[1])

        # 清除展示区
        self.ax.clear()
        # logging.info("x:{}".format(x))
        # logging.info("y:{}".format(y))
        # logging.info("x_dot:{}".format(x_dot))
        # logging.info("y_dot:{}".format(y_dot))
        # 设置箭头的形状和大小
        # 追随者
        x_temp = np.array([])
        y_temp = np.array([])
        x_temp_dot = np.array([])
        y_temp_dot = np.array([])

        # 领导者
        x_temp_f = np.array([])
        y_temp_f = np.array([])
        x_temp_dot_f = np.array([])
        y_temp_dot_f = np.array([])

        # logging.info("self.leader_list:{}".format(self.leader_list))
        for i in range(len(self.swarm)):
            if i not in list(self.leader_list):
                # x，y分别存储x,y方向上的位置
                x_temp = np.append(x_temp, self.swarm[i].pos[0])
                y_temp = np.append(y_temp, self.swarm[i].pos[1])

                # x_dot，y_dot 分别存储x,y方向上的范数
                x_temp_dot = np.append(x_temp_dot, self.swarm[i].vel[0] / norm(self.swarm[i].vel) * 0.4)
                y_temp_dot = np.append(y_temp_dot, self.swarm[i].vel[1] / norm(self.swarm[i].vel) * 0.4)

        self.ax.quiver(x_temp, y_temp, x_temp_dot, y_temp_dot, width=0.01,
                       scale=5, units="inches", color='#EC3684', angles='xy')

        for item in self.leader_list:
            # x，y分别存储x,y方向上的位置

            x_temp_f = np.append(x_temp_f, self.swarm[item].pos[0])
            # logging.info("x_temp_f:{}".format(x_temp_f))
            y_temp_f = np.append(y_temp_f, self.swarm[item].pos[1])

            # x_dot，y_dot 分别存储x,y方向上的范数
            x_temp_dot_f = np.append(x_temp_dot_f, self.swarm[item].vel[0] / norm(self.swarm[item].vel) * 0.4)
            y_temp_dot_f = np.append(y_temp_dot_f, self.swarm[item].vel[1] / norm(self.swarm[item].vel) * 0.4)
        # logging.info("x_temp_f:{}".format(x_temp_f))

        self.ax.quiver(x_temp_f, y_temp_f, x_temp_dot_f,
                       y_temp_dot_f,
                       width=0.01, scale=5, units="inches", color='#006400', angles='xy')

        # 添加画线, 画出排斥和吸引，判断是否正常运行
        for k in range(len(self.swarm)):
            # 画self.swarm[k] 与其邻居的线
            # 创建邻居集合
            neibors = []
            for m in range(len(self.swarm[k].neibour_set)):
                neibors.append([self.swarm[k].neibour_set[m].pos[0], self.swarm[k].neibour_set[m].pos[1]])
            if len(neibors) != 0:
                x_points, y_points = zip(*neibors)
                # 绘制连线
                for x, y in neibors:
                    plt.plot([self.swarm[k].pos[0], x], [self.swarm[k].pos[1], y], 'g--', linewidth=0.1, zorder=1)

        self.ax.set_aspect('auto', 'box')
        self.ax.set_xlim(0, field.width)
        self.ax.set_ylim(0, field.height)

        self.ax.tick_params(axis='x', colors='red')
        self.ax.tick_params(axis='y', colors='blue')
        circle = plt.Circle((self.target_x, self.target_y), self.target_radius, color='r', fill=False)
        plt.gcf().gca().add_artist(circle)
        plt.pause(0.01)
        # 清除个体的邻居集合
        for n in range(len(self.swarm)):
            self.swarm[n].neibour_set = []


couzin = Couzin()
for i in range(1000):
    couzin.step()
