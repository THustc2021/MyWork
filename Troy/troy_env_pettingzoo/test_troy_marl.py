import os
import tianshou as ts

from troy_env_pettingzoo.train_troy_marl import get_agents, logdir, task, get_env

if __name__ == '__main__':

    policy, agents = get_agents(load_policy=False, exploration_noise=0)

    policy.eval()
    collector = ts.data.Collector(policy, ts.env.DummyVectorEnv([lambda: get_env("human")]), exploration_noise=False)
    result = collector.collect(n_episode=1, render=1 / 35)

    # rews, lens = result["rews"], result["lens"]
    # print(f"Final reward: {rews.mean()}, length: {lens.mean()}")