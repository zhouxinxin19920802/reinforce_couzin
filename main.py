# -*- coding: utf-8 -*-
# @Author  : zhouxin
# @Time    : 2023/11/4 16:59
# @File    : main.py
# @annotation    :
import numpy as np
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
import couzin_env as env
import torch
torch.autograd.set_detect_anomaly(True)
import logging
logging.basicConfig(
    level=logging.INFO,  # 控制台打印的日志级别
    filename="test_log_0.txt",
    filemode="w",  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
    # a是追加模式，默认如果不写的话，就是追加模式
    format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    # 日志格式
)
# logger = logging.getLogger()
# logger.setLevel(logging.WARNING)



def convert_list1(obs):
    state = []
    for i in range(len(obs)):
        state_s = []
        for j in range(len(obs[i])):
            state_s = np.concatenate([state_s, obs[i][j]])
        state = np.concatenate([state, state_s])
    return state


if __name__ == '__main__':
    env_ = env.Couzin(20,0.2,Is_visual=True)
    n_agents = env_.n
    logging.info("n_agents:{}".format(n_agents))
    # 观察个体的集合
    actor_dims = [4 * n_agents] * n_agents
    # 每个个体的观察长度为 n-1
    critic_dims = n_agents  * 4 * n_agents
    # actions 先只设置为1
    # action的数量和智能体的数量一致
    n_actions = 1
    # 场景仅为测试
    scenario = "simple"
    maddpg_agents = MADDPG(env_.leader_list, actor_dims, critic_dims, n_agents, n_actions,
                           fc1=64, fc2=64,
                           alpha=0.01, beta=0.01, scenario=scenario,
                           chkpt_dir='tmp\\maddpg\\')
    memory = MultiAgentReplayBuffer(10000, critic_dims, actor_dims,
                                    n_actions, n_agents, batch_size=100)
    # 打印间隔
    PRINT_INTERVAL = 500
    N_GAMES = 100
    # steps 的次数
    MAX_STEPS = 100000
    total_steps = 0
    #
    score_history = []
    evaluate = False
    best_score = 500 

    if evaluate:
        maddpg_agents.load_checkpoint()
    for i in range(N_GAMES):
        obs = env_.reset()
        logging.info("obs:{}".format(obs))
        score = 0
        done = [False] * n_agents
        episode_step = 0
        while not any(done):
            if evaluate:
                env.is_visual = True
                # time.sleep(0.1) # to slow down the action for the video
            # logging.info("obs_before:{}".format(obs))
            actions = maddpg_agents.choose_action(obs)
            # logging.info("obs_after:{}".format(obs))
            logging.info("actions:{},{}".format(actions, len(actions)))


            obs_,  reward, done = env_.step(actions)

            logging.info("obs_:{}".format(len(obs_[0])))
            logging.info("reward:{},{}".format(reward, len(reward)))
            logging.info("done:{},{}".format(done, len(done)))

            if episode_step >= MAX_STEPS:
                done = [True] * n_agents

            # obs 和 obs_的转化

            state = env.convert_list3(obs)


            state_ = env.convert_list3(obs_)
            logging.info("state_:{}".format(state_))

            memory.store_transition(obs, state, actions, reward, obs_, state_, done)

            if total_steps % 100 == 0 and not evaluate:
                maddpg_agents.learn(memory)

            obs = obs_
            # print("reward:{}".format(reward))
            score += sum(reward)/len(reward)
            total_steps += 1
            episode_step += 1

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        # 打印运行的次数
        print("running_times:{}".format(i))
        print("avg_score:{}".format(avg_score))
        if not evaluate:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score))
