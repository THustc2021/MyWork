import math
import random
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import pprint
import os
from typing import Optional, Tuple

import gymnasium
import tianshou as ts
import numpy as np
import torch
from pettingzoo.utils.wrappers import ClipOutOfBoundsWrapper, BaseWrapper
from pettingzoo.utils.conversions import parallel_to_aec_wrapper
from tianshou.data import Collector, VectorReplayBuffer, Batch, HERVectorReplayBuffer, PrioritizedReplayBuffer, \
    PrioritizedVectorReplayBuffer
from tianshou.exploration import GaussianNoise
from tianshou.trainer import OffpolicyTrainer
from tianshou.policy import MultiAgentPolicyManager, DDPGPolicy, PPOPolicy
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from torch.utils.tensorboard import SummaryWriter

# from policy.mapolicy import MultiAgentPolicyManager
# from policy.ddpg import DDPGPolicy
from networks.netv2 import Actor, Critic, PreProcessNet
from troy_env_pettingzoo.envs.troy_env import TroyEnv
from config_values import MAX_RANGE

# model params
actor_hidden_sizes = []
for i in range(7, 12):
    actor_hidden_sizes += [2**i] * (12-i)
for i in range(11, 6, -1):
    actor_hidden_sizes += [2**i] * (12-i)
critic_hidden_sizes = []
for i in range(7, 12):
    critic_hidden_sizes += [2**i] * (12-i)
for i in range(11, 6, -1):
    critic_hidden_sizes += [2**i] * (12-i)
hidden_dim = 1024
Q_params = {"hidden_sizes": [128, 256, 512, 256, 128]}
V_params = {"hidden_sizes": [128, 256, 512, 256, 128]}

process_num = 4
#
actor_lr = 5e-4
critic_lr = 5e-4
n_step = 512    # 回望n_step步的更新
buffer_size = 102400
logdir = "results"
task = "Troy_2agents_rel_simple_award"
train_name = "autoset"

epoch = 100
step_per_epoch = 3000*2*process_num    # 一次游戏最多3000环境步，有两个智能体，每个智能体会有自己的一个step
# 一次collect收集的transitions数量（这个是结合了所有环境的统一步骤），每收集一个transition都可能（取决于下面的update_per_step）会导致一个gradient step过程
# 对于两个智能体，在一个环境step过程中会产生两个transition
# step_per_collect = 512*2*process_num    # 因此，一次更新相当于每个环境走了512步
episode_per_collect = process_num
update_per_step = 0.05  # 使用上面收集好的transitions的部分来进行梯度更新，类似于训练强度，不当大于1
batch_size = n_step*process_num   # 每次update policy使用的来自于replaybuffer的样本数量
test_num = 100

class MultiCat(torch.distributions.distribution.Distribution):
    def __init__(self, inp):
        super().__init__()
        cats = []
        ptr = 0
        for i in range(len(action_space.high[0])):
            step = action_space.high[0][i] + 1 + ptr
            cat = torch.distributions.Categorical(inp[:, :, ptr:step])
            ptr = step
            cats.append(cat)
        self.cats = cats

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        ts = []
        for cat in self.cats:
            ts.append(cat.sample(sample_shape))
        return torch.stack(ts, dim=-1)

    def log_prob(self, value):
        lps = []
        for catsi in range(len(self.cats)):
            lp = self.cats[catsi].log_prob(value[:, :, catsi])
            lps.append(lp)
        return torch.stack(lps, dim=-1)

def get_env(render_mode=None):
    return ts.env.PettingZooEnv(BaseWrapper(parallel_to_aec_wrapper(TroyEnv(render_mode=render_mode))))

# 最新的实验关掉exploration_noise，让每个环境都维护一个random state，从而提高探索效率
def get_agents(load_policy=False) :

    env = get_env()
    # 由于observation是一个复合结构，所以我们要对其进行预处理，才能交给preprocess net
    # 我们在preprocess net前再使用一个小型网络，将输入重整并归一化
    observation_space = env.observation_space
    global action_space
    action_space = env.action_space
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model
    ppprocess_net = PreProcessNet(hidden_dim, observation_space, device=device)
    net = Net(hidden_dim,
              hidden_sizes=actor_hidden_sizes,
              device=device)
    actor = Actor(ppprocess_net, net, action_space.shape or action_space.n,
                  max_actions=action_space.high[0], device=device).to(device)
    ppprocess_net = PreProcessNet(hidden_dim, observation_space, device=device)
    net = Net(
        hidden_dim,
        action_space.shape or action_space.n,
        hidden_sizes=critic_hidden_sizes,
        concat=False,
        dueling_param=(Q_params, V_params), # 使用dueling dqn
        device=device,
    )
    critic = Critic(ppprocess_net, net, device=device).to(device)
    optimizer = torch.optim.AdamW([
        {'params': actor.parameters(), 'lr': actor_lr},
        {'params': critic.parameters(), 'lr': critic_lr},
    ])
    agent_troy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optimizer,
        action_space=env.action_space,
        dist_fn=MultiCat,
        action_scaling=False,
        action_bound_method=None,
        lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-5)
    )
    # agent_troy = RandomPolicy(action_space=env.action_space)

    # model
    ppprocess_net = PreProcessNet(hidden_dim, observation_space, device=device)
    net = Net(hidden_dim,
              hidden_sizes=actor_hidden_sizes,
              device=device)
    actor = Actor(ppprocess_net, net, action_space.shape or action_space.n,
                  max_actions=action_space.high[0], device=device).to(device)
    ppprocess_net = PreProcessNet(hidden_dim, observation_space, device=device)
    net = Net(
        hidden_dim,
        action_space.shape or action_space.n,
        hidden_sizes=critic_hidden_sizes,
        concat=False,
        dueling_param=(Q_params, V_params), # 使用dueling dqn
        device=device,
    )
    critic = Critic(ppprocess_net, net, device=device).to(device)
    agent_greek = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optimizer,
        action_space=env.action_space,
        dist_fn=MultiCat,
        action_scaling=False,
        action_bound_method=None,
        lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-5)
    )
    # agent_greek = RandomPolicy(action_space=env.action_space)

    agents = [agent_troy, agent_greek]
    policy = MultiAgentPolicyManager(policies=agents, env=env, action_scaling=False, action_bound_method=None)

    global train_name
    train_name = f"{agent_troy._get_name()}_{agent_greek._get_name()}"

    if load_policy:
        # 加载模型参数
        print(f"load pretrained policies: {os.path.join(logdir, task, f'{train_name}_pth')}")
        model_troy_save_path = os.path.join(logdir, task, f"{train_name}_pth", "troy_policy_1.pth")
        model_greek_save_path = os.path.join(logdir, task, f"{train_name}_pth", "greek_policy_1.pth")
        policy.policies[env.agents[0]].load_state_dict(torch.load(model_troy_save_path))
        policy.policies[env.agents[1]].load_state_dict(torch.load(model_greek_save_path))

        policy.policies[env.agents[0]].actor.is_training = False
        policy.policies[env.agents[1]].actor.is_training = False

        # 冻结参数
        # policy.policies[env.agents[0]].actor.requires_grad_(False)

    return policy, env.agents

if __name__ == '__main__':

    train_envs = ts.env.SubprocVectorEnv([lambda: get_env() for _ in range(process_num)])
    # train_envs = ts.env.DummyVectorEnv([lambda: get_env() for _ in range(1)])
    # test_envs = ts.env.DummyVectorEnv([lambda: _get_env() for _ in range(4)]) # 不做测试

    # seed
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_envs.seed(seed)
    # test_envs.seed(seed)

    # ======== Step 2: Agent setup =========
    policy, agents = get_agents(load_policy=False)

    # ======== Step 3: Collector setup =========
    # collector
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(buffer_size, len(train_envs)),
        # PrioritizedVectorReplayBuffer(buffer_size, len(train_envs), alpha=0.6, beta=0.3),
        exploration_noise=False,
    )
    # test_collector = Collector(policy, test_envs)
    test_collector = None
    # log
    log_path = os.path.join(logdir, task, train_name)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    # ======== Step 4: Callback functions setup =========
    def reward_metric(rewards):
        return np.max(rewards, axis=1)  # 返回两个智能体奖励中最大的那个

    def save_best_fn(policy):
        print("best_policy saved.")
        model_troy_save_path = os.path.join(logdir, task, f"{train_name}_pth", "troy_policy.pth")
        model_greek_save_path = os.path.join(logdir, task, f"{train_name}_pth", "greek_policy.pth")
        os.makedirs(os.path.join(logdir, task, f"{train_name}_pth"), exist_ok=True)
        torch.save(policy.policies[agents[0]].state_dict(), model_troy_save_path)
        torch.save(policy.policies[agents[1]].state_dict(), model_greek_save_path)

    def save_checkpoint_fn(epoch: int, env_step: int, gradient_step: int):
        print("save current state_dict.")
        model_troy_save_path = os.path.join(logdir, task, f"{train_name}_pth", f"troy_policy_{epoch}.pth")
        model_greek_save_path = os.path.join(logdir, task, f"{train_name}_pth", f"greek_policy_{epoch}.pth")
        os.makedirs(os.path.join(logdir, task, f"{train_name}_pth"), exist_ok=True)
        torch.save(policy.policies[agents[0]].state_dict(), model_troy_save_path)
        torch.save(policy.policies[agents[1]].state_dict(), model_greek_save_path)
        # 衰减探索噪声
        if policy.policies[agents[0]].actor.random_state > 0.05:
            policy.policies[agents[0]].actor.random_state = 0.5 * policy.policies[agents[0]].actor.random_state
            policy.policies[agents[1]].actor.random_state = 0.5 * policy.policies[agents[1]].actor.random_state
            print(f"reset random state to {policy.policies[agents[1]].actor.random_state}")
        return os.path.join(logdir, task, f"{train_name}_pth")

    # ======== Step 5: Run the trainer =========
    # trainer
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=epoch,
        step_per_epoch=step_per_epoch,
        # step_per_collect=step_per_collect,
        episode_per_collect=episode_per_collect,
        episode_per_test=test_num,
        batch_size=batch_size,
        update_per_step=update_per_step,
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
        logger=logger,
        test_in_train=False
    ).run()

    print(f"\n==========Result==========\n{result}")
    print("\n(the trained policy can be accessed via policy.policies)")

    pprint.pprint(result)
