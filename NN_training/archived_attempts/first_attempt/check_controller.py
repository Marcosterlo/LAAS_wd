import argparse
from stable_baselines3 import PPO
from pendulum_env import Pendulum_env
import numpy as np
import matplotlib.pyplot as plt

env = Pendulum_env()

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str)
args = parser.parse_args()
model = PPO.load(args.model_name)

nstep = 500
state = []
u = []


obs = env.reset()
print(obs[0])
cazz = 0
for i in range(nstep):
    if i == 0:
        action, _states = model.predict(obs[0], deterministic=True)
    else:
        action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action[0])
    cazz += reward
    state.append(obs)
    u.append(action[0])

print(cazz)

state = np.array(state)
u = np.array(u)
time_grid = np.linspace(0, nstep, nstep)

plt.plot(time_grid, state[:,0])
plt.grid(True)
plt.title("Position")
plt.show()

plt.plot(time_grid, state[:,1])
plt.grid(True)
plt.title("Velocity")
plt.show()

plt.plot(time_grid, u)
plt.grid(True)
plt.title("Control")
plt.show()