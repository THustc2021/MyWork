import gymnasium
import os
from gymnasium.envs.registration import register
os.chdir("..")

register(
    id='Troy-v0',
    entry_point='troy_env_gym.envs.troy:TroyEnv',
)

env = gymnasium.make("Troy-v0", render_mode="human")

observation, info = env.reset()

for _ in range(10000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
