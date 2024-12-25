import os
os.chdir("..")

from troy_env_pettingzoo.envs.troy_env import TroyEnv

env = TroyEnv(render_mode="human")
observations, infos = env.reset()

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
    # print(rewards)
    if True in terminations.values() or True in truncations.values():
        break

env.close()