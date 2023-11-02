import numpy as np
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
from make_env import make_env
import logging

logging.basicConfig(
    level=logging.DEBUG,  # 控制台打印的日志级别
    filename="test_log.txt",
    filemode="w",  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
    # a是追加模式，默认如果不写的话，就是追加模式
    format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    # 日志格式
)
# 将日志级别设置为WARNING，这将关闭INFO和DEBUG级别的日志

# logging开关
logger = logging.getLogger()
logger.setLevel(logging.WARNING)


# Array合并为list
def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

if __name__ == '__main__':
    #scenario = 'simple'
    scenario = 'simple_adversary'
    env = make_env(scenario)
    n_agents = env.n
    actor_dims = []
    # actor的输入，其它智能体的观察空间集合
    for i in range(n_agents):
        actor_dims.append(env.observation_space[i].shape[0])
    logging.info("actor_dims:".format(actor_dims))
    critic_dims = sum(actor_dims)
    logging.info("critic_dims:".format(critic_dims))
    
    # action space is a list of arrays, assume each agent has same action space
    n_actions = env.action_space[0].n

    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                           fc1=64, fc2=64,  
                           alpha=0.01, beta=0.01, scenario=scenario,
                           chkpt_dir='tmp\\maddpg\\')

    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, 
                        n_actions, n_agents, batch_size=10)

    PRINT_INTERVAL = 500
    N_GAMES = 2000
    MAX_STEPS = 25
    total_steps = 0
    score_history = []
    evaluate = True
    best_score = -85

    if evaluate:
        maddpg_agents.load_checkpoint()

    for i in range(N_GAMES):
        obs = env.reset()
        score = 0
        done = [False]*n_agents
        episode_step = 0
        while not any(done):
            if evaluate:
                env.render()
                #time.sleep(0.1) # to slow down the action for the video
            actions = maddpg_agents.choose_action(obs)
            obs_, reward, done, info = env.step(actions)
            logging.info("obs_:{}".format(obs_))
            logging.info("actions:{}".format(actions))
            logging.info("reward:{}".format(reward))

            state = obs_list_to_state_vector(obs)
            logging.info("state:{}".format(len(state)))
            state_ = obs_list_to_state_vector(obs_)

            if episode_step >= MAX_STEPS:
                done = [True]*n_agents

            memory.store_transition(obs, state, actions, reward, obs_, state_, done)

            if total_steps % 100 == 0 and not evaluate:
                maddpg_agents.learn(memory)

            obs = obs_

            score += sum(reward)
            total_steps += 1
            episode_step += 1

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if not evaluate:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score))
