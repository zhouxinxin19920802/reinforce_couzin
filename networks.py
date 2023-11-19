import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

import logging
logging.basicConfig(
    level=logging.DEBUG,  # 控制台打印的日志级别
    filename="test_log_0.txt",
    filemode="w",  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
    # a是追加模式，默认如果不写的话，就是追加模式
    format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    # 日志格式
)


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, 
                    n_agents, n_actions, name, chkpt_dir):
        super(CriticNetwork, self).__init__()

        # print("name:",name)
        self.chkpt_file = os.path.join(chkpt_dir, name)
        # print("self.chkpt_file:",self.chkpt_file)
        

        # logging.info("input_dims+n_agents*n_actions:{}".format(input_dims+n_agents*n_actions))
        # logging.info("input_dims+n_agents*n_actions:{}".format(input_dims+n_agents*n_actions))
        # 
        self.fc1 = nn.Linear(input_dims+n_agents*n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state, action):
        logging.info("T.cat([state, action]:{}:".format(T.cat([state, action], dim=1)))
        x1 = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x2 = F.relu(self.fc2(x1))
        q = self.q(x2)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))

# 可视角的actor网络
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, 
                 n_actions, name, chkpt_dir):
        super(ActorNetwork, self).__init__()
        # print("name:",name)
        self.chkpt_file = os.path.join(chkpt_dir, name)
        # print("self.chkpt_file:",self.chkpt_file)

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state):
        x1 =  F.relu(self.fc1(state),inplace=False)
        x2 =  F.relu(self.fc2(x1),inplace=False)
        pi1 = F.relu(self.fc3(x2),inplace=False)

        return pi1

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


# 目标影响权重的actor网络
class ActorNetwork_w(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, 
                 n_actions, name, chkpt_dir):
        super(ActorNetwork_w, self).__init__()
        # print("name:",name)
        self.chkpt_file = os.path.join(chkpt_dir, name)
        # print("self.chkpt_file:",self.chkpt_file)

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state):
        x1 =  F.relu(self.fc1(state),inplace=False)
        x2 =  F.relu(self.fc2(x1),inplace=False)
        pi1 = F.relu(self.fc3(x2),inplace=False)
        cons = T.sigmoid(pi1)
        return cons

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


