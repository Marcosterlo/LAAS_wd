from NN_training.models.NeuralNet_simple import NeuralNet
import torch
import torch.optim as optim
import torch.nn as nn
from stable_baselines3 import PPO
from NN_training.environments.Nn_3l_env import Nn_3l_env

env = Nn_3l_env()

policy_kwargs = dict(
    net_arch=[32, 32, 32],
    activation_fn=nn.Tanh
)
model_rl = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, n_steps=4, verbose=1)

model_rl.learn(total_timesteps=10000)