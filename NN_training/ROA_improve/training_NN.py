import torch.nn as nn
from stable_baselines3 import PPO
from NN_training.environments.Nn_3l_env import Nn_3l_env
import numpy as np

env = Nn_3l_env()

policy_kwargs = dict(
    net_arch=[512, 1024, 1024, 2048],
    activation_fn=nn.ReLU
)
model_rl = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, n_steps=64, batch_size=64, verbose=1)

model_rl.learn(total_timesteps=300)

for i, weight in enumerate(env.last_W):
    np.save('W' + str(i) + '.npy', weight)
for i, bias in enumerate(env.last_b):
    np.save('b' + str(i) + '.npy', bias)