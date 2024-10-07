from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from pendulum_env import Pendulum_env
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

env = Pendulum_env()
env = Monitor(env, filename='./monitor_logs')

check_env(env, warn=True)

policy_kwargs = dict(
    net_arch=[32, 32, 32, 32],
    activation_fn=nn.Tanh
)

model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, ent_coef=0.05, gamma=0.995, learning_rate=1e-4, clip_range=0.1, n_steps=4096)

model.learn(total_timesteps=500000)

nstep = 1000
x0 = np.array([np.pi/6, 0]).astype(np.float32)

state = []
actions = []
obs = env.reset()[0]

for _ in range(nstep):
    action, _ = 0*model.predict(obs)
    obs, _, done, _, _ = env.step(action[0])
    state.append(obs)
    actions.append(action)
    if done:
        print("==== OUTSIDE OF BOUNDS ====")
        obs = env.reset()[0]

states = np.array(state)
actions = np.array(actions)

time_grid = np.linspace(0, nstep, nstep)
fig, axs = plt.subplots(3)
axs[0].plot(time_grid, states[:, 0])
axs[0].grid(True)
axs[0].set_xlabel("Steps")
axs[0].set_ylabel("Position")
axs[1].plot(time_grid, states[:, 1])
axs[1].grid(True)
axs[1].set_xlabel("Steps")
axs[1].set_ylabel("Velocity")
axs[2].plot(time_grid, actions)
axs[2].grid(True)
axs[2].set_xlabel("Steps")
axs[2].set_ylabel("Control u")
plt.show()