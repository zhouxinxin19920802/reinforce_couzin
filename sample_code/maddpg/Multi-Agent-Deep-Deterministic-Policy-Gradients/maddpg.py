import torch as T
import torch.nn.functional as F
from agent import Agent
import logging

logging.basicConfig(
    level=logging.DEBUG,  # 控制台打印的日志级别
    filename="test_log.txt",
    filemode="w",  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
    # a是追加模式，默认如果不写的话，就是追加模式
    format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    # 日志格式
)

class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, 
                 scenario='simple',  alpha=0.01, beta=0.01, fc1=64, 
                 fc2=64, gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/'):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario 
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,  
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

        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        device = self.agents[0].actor.device

        # 所有个体的状态,所有个体的动作,所有个体的奖励(共享奖励)
        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)
        logging.info("states:{},{}".format(states,len(states)))
        logging.info("actions:{},{}".format(actions,len(actions)))
        logging.info("rewards:{},{}".format(rewards,len(rewards)))
        logging.info("dones:{},{}".format(dones,len(dones)))

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            # 新状态
            new_states = T.tensor(actor_new_states[agent_idx], 
                                 dtype=T.float).to(device)
            # 预测的新动作
            new_pi = agent.target_actor.forward(new_states)

            # 新动作集合
            all_agents_new_actions.append(new_pi)

            # 当前状态
            mu_states = T.tensor(actor_states[agent_idx], 
                                 dtype=T.float).to(device)
            # 当前状态下预测的动作
            pi = agent.actor.forward(mu_states)
            # 当前状态下预测的动作的集合
            all_agents_new_mu_actions.append(pi)
            # 
            old_agents_actions.append(actions[agent_idx])
        logging.info("all_agents_new_actions:{},{}".format(all_agents_new_actions,len(all_agents_new_actions)))
        logging.info("all_agents_new_mu_actions:{},{}".format(all_agents_new_mu_actions,len(all_agents_new_mu_actions)))
        logging.info("rewards:{},{}".format(rewards,len(rewards)))


        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions],dim=1)

        logging.info("new_actions:{},{}".format(new_actions,len(new_actions)))
        logging.info("mu:{},{}".format(mu,len(mu)))
        logging.info("old_actions :{},{}".format(old_actions,len(old_actions)))
        
        # 对每个Agents的参数进行更新
        for agent_idx, agent in enumerate(self.agents):
            logging.info("agent.target_critic.forward(states_, new_actions):{}".format(agent.target_critic.forward(states_, new_actions)))
            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
            logging.info("critic_value_:{}".format(critic_value_))
            critic_value_[dones[:,0]] = 0.0
            critic_value = agent.critic.forward(states, old_actions).flatten()

            target = rewards[:,agent_idx] + agent.gamma*critic_value_
            critic_loss = F.mse_loss(target, critic_value)
            logging.info("critic_loss:{}".format(critic_loss))
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -T.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

            agent.update_network_parameters()
