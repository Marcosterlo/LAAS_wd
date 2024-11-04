import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from NN_training.environments.simple_pendulum_env import Simple_pendulum_env
import numpy as np
from NN_training.models.NeuralNet_simple import NeuralNet
import torch

env = Simple_pendulum_env()

deepl_model = NeuralNet()

state_dict = torch.load('model.pth', map_location=torch.device('cpu'))

deepl_model.load_state_dict(state_dict)
weights = {name: param.data.numpy() for name, param in deepl_model.named_parameters()}

class CustomPolicy(ActorCriticPolicy):
  def __init__(self, *args, **kwargs):
    super(CustomPolicy, self).__init__(*args, **kwargs)
    
    for name, param in self.named_parameters():
      param.data = torch.tensor(weights[name])

model = PPO(CustomPolicy, env, verbose=1)