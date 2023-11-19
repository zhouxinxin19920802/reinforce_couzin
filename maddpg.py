import torch as T
import torch.nn.functional as F
from agent import Agent
import logging
import torch
from networks import ActorNetwork
from networks import ActorNetwork_w
# from main import env_

torch.autograd.set_detect_anomaly(True)
# logging.basicConfig(
#     level=logging.DEBUG,  # 控制台打印的日志级别
#     filename="test_log.txt",
#     filemode="w",  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
#     # a是追加模式，默认如果不写的话，就是追加模式
#     format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s"
#     # 日志格式
# )


class MADDPG:
    def __init__(self, leader_list ,actor_dims, critic_dims, n_agents, n_actions,
                 scenario='simple', alpha=0.01, beta=0.01, fc1=66,
                 fc2=67, gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/'):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario

        # Agent里面包含了 各个智能体里的网络初始
        # 各个智能体的网络的 初始化，参数更新，模型保存
        # for agent_idx in range(self.n_agents):
        #     self.agents.append(Agent(actor_dims[agent_idx], critic_dims,
        #                              n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
        #                              chkpt_dir=chkpt_dir))

        # 增加一个功能,actor网络和共享，先创建第一个actor网络，然后再向其它actor网络共享
        # 直接创建actor网络和target actor网络进行传入
        # 需要在更新时候同步更新
        ####################################################################  
        # 网络创建
        # 直接输入actor维数
        actor_dims = n_agents  * 4
        fc1 = 64
        fc2 = 64
        n_actions = 1
        alpha=0.01
        # general_actor_name,general_actor_name 创建时候传入名字
        general_actor_name = "general_actor"
        general_target_actor_name = "general_actor_name"
        general_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, 
                                  chkpt_dir=chkpt_dir,  name=general_actor_name) 
        general_target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, 
                                  chkpt_dir=chkpt_dir,  name=general_target_actor_name) 
        
        general_actor_name_leader = "general_actor_leader"
        general_target_actor_name_leader = "general_actor_name_leader"
        general_actor_leader = ActorNetwork_w(alpha, actor_dims, fc1, fc2, n_actions, 
                                  chkpt_dir=chkpt_dir,  name=general_actor_name_leader) 
        general_target_actor_leader = ActorNetwork_w(alpha, actor_dims, fc1, fc2, n_actions, 
                                  chkpt_dir=chkpt_dir,  name=general_target_actor_name_leader) 

        # 传入每个agent里面
        logging.info("chkpt_dir:{}".format(chkpt_dir))
        for agent_idx in range(self.n_agents):
            # 加一个判断, 判断该个体是否为领导者,领导者要传入目标
            if agent_idx in leader_list:
                self.agents.append(Agent(general_actor_leader,  general_target_actor_leader, critic_dims,
                                            n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
                                            chkpt_dir=chkpt_dir))
            # 增加
            else:
                   self.agents.append(Agent(general_actor, general_target_actor, critic_dims,
                                            n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
                                            chkpt_dir=chkpt_dir))

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx])
            actions.append(action)
        return actions

    def learn(self, memory):
        if not memory.ready():
            return

        print("learing")
        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()
        logging.info("actor_states:{}".format(states))
        device = self.agents[0].actor.device

        # 所有个体的状态,所有个体的动作,所有个体的奖励(共享奖励)
        states =  T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)
        logging.info("states:{},{}".format(states, len(states)))
        logging.info("actions:{},{},{}".format(actions, len(actions), len(actions[0])))
        logging.info("rewards:{},{}".format(rewards, len(rewards)))
        logging.info("dones:{},{}".format(dones, len(dones)))

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            # 新状态
            new_states = T.tensor(actor_new_states[agent_idx],
                                  dtype=T.float).to(device)
            # logging.info("new_states:{}".format(new_states))
            # 预测的新动作
            new_pi = agent.target_actor.forward(new_states)
            logging.info("new_pi:{}".format(new_pi))
            # 新动作集合
            all_agents_new_actions.append(new_pi)

            # 当前状态
            mu_states = T.tensor(actor_states[agent_idx],
                                 dtype=T.float).to(device)
            # 当前状态下预测的动作
            pi = agent.actor.forward(mu_states)
            logging.info("pi:{}".format(pi))
            # 当前状态下预测的动作的集合
            all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(actions[agent_idx])

        logging.info("all_agents_new_actions:{},{}".format(all_agents_new_actions, len(all_agents_new_actions)))
        logging.info(
            "all_agents_new_mu_actions:{},{}".format(all_agents_new_mu_actions, len(all_agents_new_mu_actions)))
        logging.info("rewards:{},{}".format(rewards, len(rewards)))
        # 记住这个事，all_agents_new_actions横坐标是个体，每个个体里，包含了指定的样本数，如果10个样本， 1*10 1*10 1*10, T.cat是把
        # 每个个体的第一个样本给抽出来合并在一起，就是那个例子下，所有个体的动作值
        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions], dim=1)

        # Tcat 用于合并 15 * 1024将所有动作合并起来
        logging.info("new_actions:{},{}".format(new_actions, len(new_actions)))
        logging.info("mu:{},{}".format(mu, len(mu)))
        logging.info("old_actions :{},{}".format(old_actions, len(old_actions)))

        # 对每个Agents的参数进行更新
        for agent_idx, agent in enumerate(self.agents):
            # 针对每个个体进行了网络更新
            # logging.info("agent.target_critic.forward(states_, new_actions):{}".format(
            #     agent.target_critic.forward(states_, new_actions)))
            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
            # done是重复的
            logging.info("critic_value_:{}".format(critic_value_))
            logging.info("dones[:,0]:{}".format(dones[:, 0]))
            # 取第一个智能体状态
            critic_value_[dones[:, 0]] = 0.0
            logging.info("critic_value_:{}".format(critic_value_))
            critic_value = agent.critic.forward(states, old_actions).flatten()
            logging.info("critic_value:{}".format(critic_value))
            #  rewards[:,agent_idx] 这是取第一个第一列的reward
            logging.info("rewards[:, agent_idx]:{}".format(rewards[:, agent_idx]))
            target = rewards[:, agent_idx] + agent.gamma * critic_value_
            logging.info("target:{}".format(target))
            critic_loss = F.mse_loss(target, critic_value)
            logging.info("critic_loss:{}".format(critic_loss))
            agent.critic.optimizer.zero_grad()
            # with torch.autograd.set_detect_anomaly(True):
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()
            logging.info("training")
            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -T.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()
            agent.update_network_parameters()

            # 如果直接更新的话，循环对每个个体进行更新的时候，每次actor和target_actor都会被更新
