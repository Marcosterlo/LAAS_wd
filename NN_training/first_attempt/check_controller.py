from stable_baselines3 import PPO
from pendulum_env import Pendulum_env
import numpy as np
import matplotlib.pyplot as plt

env = Pendulum_env()
model = PPO.load('better_one.zip')

state = []
u = []

obs = env.reset()
for i in range(3000):
    if i == 0:
        action, _states = model.predict(obs[0], deterministic=True)
    else:
        action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action[0]*0)
    state.append(obs)
    u.append(action[0])