import numpy as np
from numpy.linalg import *
from math import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random


# 红色 x轴，蓝 y轴，绿色 z轴
dimension = '3d'    # 2d/3d
n = 170             # number of agents
dt = 0.1
# r_r = 1
# r_o = 10
# r_a = 20

# 最小距离设置 α
a_minimal_range = 1.8



field_of_view = 3*pi/2
theta_dot_max = 1
constant_speed = 3.5
np.seterr(divide='ignore', invalid='ignore')
break_r = 0

# g 偏好速度方向
g = np.array([-4, 5, -1])
g = g/norm(g)

# w 设置影响权重
w_p = 1.2

# p 设置领导者比例
p = 0.1


class Field:
    def __init__(self):
        self.width = 500    # x_max[m]
        self.height = 500   # y_max[m]
        self.depth = 500    # z_max[m]


def cal_angle_of_vector(v0, v1):
    dot_product = np.dot(v0, v1)
    v0_len = np.linalg.norm(v0)
    v1_len = np.linalg.norm(v1)
    try:
        angle_rad = np.arccos(dot_product / (v0_len * v1_len))
    except ZeroDivisionError as error:
        raise ZeroDivisionError("{}".format(error))
    return angle_rad



class Agent:
    def __init__(self, agent_id, speed):
        self.id = agent_id
        self.pos = np.array([0, 0, 0])
        self.pos[0] = np.random.uniform(field.width*1/3, field.width*2/3)
        self.pos[1] = np.random.uniform(field.height*1/3, field.height*2/3)
        self.pos[2] = np.random.uniform(field.depth*1/3, field.depth*2/3)
        self.vel = np.random.uniform(-5, 5, 3)
        if dimension == '2d':
            self.pos[2] = 0
            self.vel[2] = 0
        # 各个方向的速度向量
        self.vel = self.vel / norm(self.vel) * speed

       # 是否被选为领导
        self.is_leader = False

    def update_position(self, delta_t):
        self.pos = self.pos + self.vel * delta_t


def rotation_matrix_about(axis, theta):
    axis = np.asarray(axis)
    axis = axis / sqrt(np.dot(axis, axis))
    a = cos(theta / 2.0)
    b, c, d = -axis * sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])



def onclick(event):
    global break_r
    break_r = 1


def get_n_rand(n,p):
    leader_list = set()
    while True:
        leader_list.add(random.randint(0, n-1))
        if len(leader_list) == n * p:
            break

    return  leader_list





if __name__ == '__main__':

    swarm = []
    field = Field()
    [swarm.append(Agent(i, constant_speed)) for i in range(n)]

    # 随机选取一定比例的点作为领导者，并生成领导编号列表
    leader_list = get_n_rand(n,p)

    print("leader_list:",leader_list)

    for i in range(len(swarm)):
        for leader_id in leader_list:
            if i == leader_id:
                swarm[i].is_leader = True

    fig = plt.figure()
    if dimension == '3d':
        ax = fig.add_axes(Axes3D(fig))
    else:
        ax = fig.gca()
    cid = fig.canvas.mpl_connect('close_event', onclick)

    with open("data.txt", "w") as f:
           f.write("{}:{}\n".format("数量",n))
           f.write("{}:{}\n".format("时间步长",dt))
           f.write("{}:{}\n".format("排斥半径", a_minimal_range))
           f.write("{}:{}\n".format("速度偏好权重",w_p))
           f.write("{}:{}\n".format("领导者比例",p))
           f.write("{}:{}\n".format("速度",constant_speed))
           f.write("{}:{}\n".format("最大角速度",theta_dot_max))
           f.write("{}:{}\n".format("偏好方向", g))
           f.write("{}:{:.2f}\n".format("视野范围", field_of_view))
           f.write("{}:{}\n".format("leader_list", leader_list))

    # 写表头
    with open("data.txt", "a+") as f:
           f.write("{}\t,{}\t,{}\t,{}\t,{}\t,{}\t,{}\t,{}\t,{}\t\n".format("帧数","编号","位置x","位置y","位置z","速度","速度x分量","速度y分量","速度z分量"))
    for step in range(1000):
        if break_r==1:
            break
        x = np.array([])
        y = np.array([])
        z = np.array([])
        x_dot = np.array([])
        y_dot = np.array([])
        z_dot = np.array([])
        for agent in swarm:
            # x，y,z 分别存储x,y,z的位置
            x = np.append(x, agent.pos[0])
            y = np.append(y, agent.pos[1])
            z = np.append(z, agent.pos[2])
            # x_dot，y_dot,z_dot 分别存储x,y,z上的速度方向
            x_dot = np.append(x_dot, agent.vel[0])
            y_dot = np.append(y_dot, agent.vel[1])
            z_dot = np.append(z_dot, agent.vel[2])

        ax.clear()
        if dimension == '2d':

            # 设置箭头的形状和大小
            x_temp = np.array([])
            y_temp = np.array([])
            z_temp = np.array([])

            x_temp_dot = np.array([])
            y_temp_dot = np.array([])
            z_temp_dot = np.array([])

            for i in range(len(swarm)):
                for leader_id in list(leader_list):
                    if i == leader_id:
                        continue
                    # x，y分别存储x,y,z方向上的位置
                    x_temp = np.append(x_temp, swarm[i].pos[0])
                    y_temp = np.append(y_temp, swarm[i].pos[1])

                    # x_dot，y_dot 分别存储x,y,z方向上的范数
                    x_temp_dot = np.append(x_temp_dot, swarm[i].vel[0])
                    y_temp_dot = np.append(y_temp_dot, swarm[i].vel[1])

            ax.quiver(x_temp, x_temp,  x_temp_dot /80 * field.width, y_temp_dot /80 * field.width,width=0.02, scale=50,
                      units="inches",color='#EC3684')

            for leader_id in list(leader_list):
                ax.quiver(x[leader_id], y[leader_id], swarm[leader_id].vel[0] /80* field.width, swarm[leader_id].vel[1] /80* field.width,
                          width=0.02, scale=50, units="inches",color='#006400')


            ax.set_aspect('auto', 'box')
            ax.set_xlim(0, field.width)
            ax.set_ylim(0, field.height)

            ax.tick_params(axis='x', colors='red')
            ax.tick_params(axis='y', colors='blue')

        else:
            # 设置箭头的形状和大小
            x_temp= np.array([])
            y_temp = np.array([])
            z_temp= np.array([])

            x_temp_dot = np.array([])
            y_temp_dot = np.array([])
            z_temp_dot = np.array([])
            # print("leader_list:",leader_list)
            for i in range(len(swarm)):
                 if i not in leader_list:

                    # x，y,z 分别存储x,y,z方向上的绝对速度
                    x_temp = np.append(x_temp,  swarm[i].pos[0])
                    y_temp  = np.append(y_temp, swarm[i].pos[1])
                    z_temp = np.append(z_temp,  swarm[i].pos[2])
                    # x_dot，y_dot,z_dot 分别存储x,y,z方向上的范数
                    x_temp_dot = np.append(x_temp_dot, swarm[i].vel[0])
                    y_temp_dot = np.append(y_temp_dot, swarm[i].vel[1])
                    z_temp_dot = np.append(z_temp_dot, swarm[i].vel[2])
            # print(len(x_temp),len(y_temp),len(z_temp),len(x_temp_dot),len(y_temp_dot),len(z_temp_dot))

            ax.quiver(x_temp, y_temp, z_temp, x_temp_dot / 80 * field.width, y_temp_dot / 80 * field.width, z_temp_dot / 80 * field.width, color='#EC3684')
            for leader_id in list(leader_list):
                ax.quiver(x[leader_id], y[leader_id], z[leader_id],  swarm[leader_id].vel[0] / 80 * field.width, swarm[leader_id].vel[1] / 80 * field.width,  swarm[leader_id].vel[2] / 80 * field.width,
                          color='#006400')
            ax.set_aspect('equal', 'box')
            ax.set_xlim(0, field.width)
            ax.set_ylim(0, field.height)
            ax.set_zlim(0, field.depth)

            ax.tick_params(axis='x', colors='red')
            ax.tick_params(axis='y', colors='blue')
            ax.tick_params(axis='z', colors='green')


        plt.pause(0.1)

        for agent in swarm:
#       2002年couzin经典模型
            # d = 0
            # d_r = 0
            # d_o = 0
            # d_a = 0
            # for neighbor in swarm:
            #     if agent.id != neighbor.id:
            #         r = neighbor.pos - agent.pos
            #         r_normalized = r/norm(r)
            #         norm_r = norm(r)
            #         agent_vel_normalized = agent.vel/norm(agent.vel)
            #         if acos(np.dot(r_normalized, agent_vel_normalized)) < field_of_view / 2:
            #             if norm_r < r_r:
            #                 d_r = d_r - r_normalized
            #             elif norm_r < r_o:
            #                 d_o = d_o + neighbor.vel/norm(neighbor.vel)
            #             elif norm_r < r_a:
            #                 d_a = d_a + r_normalized
            # if norm(d_r) != 0:
            #     d = d_r
            # elif norm(d_a) != 0 and norm(d_o) != 0:
            #     d = (d_o + d_a)/2
            # elif norm(d_a) != 0:
            #     d = d_o
            # elif norm(d_o) != 0:
            #     d = d_a
#**************************************************************
#       2005 couzin领导模型
            d = 0
            dr = 0
            da = 0

            dv = agent.vel
            for neighbor in swarm:
                if agent.id != neighbor.id:
                    # 位置向量，单位位置向量，距离
                    r = neighbor.pos - agent.pos
                    r_normalized = r / norm(r)
                    norm_r = norm(r)
                    # 速度向量
                    agent_vel_normalized = agent.vel/norm(agent.vel)
                    if acos(np.dot(r_normalized, agent_vel_normalized)) < field_of_view / 2:
                        if norm_r < a_minimal_range:
                            dr = dr - r_normalized
                        else:
                            da = da + r_normalized
                            dv = dv + neighbor.vel/norm(neighbor.vel)
            # print("da:",da)
            if norm(dr) != 0:
                if agent.is_leader == True:
                    dr = dr / norm(dr)
                    d  = (dr + w_p * g) / norm(dr + w_p * g)
                else:
                    d = dr / norm(dr)
            elif norm(da) != 0:
                if agent.is_leader == True:

                    d_new = (da + dv) / norm( da + dv )
                    d = (d_new + w_p* g)/ norm(d_new + w_p * g)
                else:

                    d_new = (da + dv) / norm(da + dv)
                    d = d_new

            if norm(d) != 0:
                z = np.cross(d/norm(d), agent.vel/norm(agent.vel))
                angle_between=cal_angle_of_vector(d,agent.vel)
                if angle_between >= theta_dot_max*dt:
                    rot = rotation_matrix_about(z, theta_dot_max*dt)
                    vel0 = np.asmatrix(agent.vel) * rot
                    vel0 = np.asarray(vel0)[0]

                    rot1 = rotation_matrix_about(z, -theta_dot_max * dt)
                    vel1 = np.asmatrix(agent.vel) * rot1
                    vel1 = np.asarray(vel1)[0]

                    if cal_angle_of_vector(vel0,d)<cal_angle_of_vector(vel1,d):
                        agent.vel=vel0
                    else:
                        agent.vel=vel1
                else:
                    agent.vel = d/norm(d) * constant_speed
                    # print(" agent.vel2:", agent.vel)
            # print("agent_vel",agent.vel)
            # print("agent_vel_norm", norm(agent.vel))
        [agent.update_position(dt) for agent in swarm]
        # id,位置x、y、z，速度大小norm(vel)，速度方向vel[0],vel[1],vel[2],是否是领导

        with open("data.txt", "a+") as f:
            for sinlge_swarm in swarm:
                f.write("{}\t,{:.2f}\t,{:.2f}\t,{:.2f}\t,{:.2f}\t,{:.2f}\t,{:.2f}\t,{:.2f}\t,{:.2f}\t\n".format(step, sinlge_swarm.id,sinlge_swarm.pos[0],sinlge_swarm.pos[1],
                                                                 sinlge_swarm.pos[2],norm(sinlge_swarm.vel),sinlge_swarm.vel[0],
                                                                 sinlge_swarm.vel[1],sinlge_swarm.vel[2]))



