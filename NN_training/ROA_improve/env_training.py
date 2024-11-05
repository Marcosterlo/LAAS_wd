from stable_baselines3 import PPO
from NN_training.environments.simple_pendulum_env import Simple_pendulum_env
from stable_baselines3.common.callbacks import BaseCallback
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env = Simple_pendulum_env()

policy_kwargs = dict(activation_fn=torch.nn.Hardtanh,
                     net_arch=[32, 32, 32])

model_rl = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

class CustomCallback(BaseCallback):
  def __init__(self, verbose=0):
    super().__init__(verbose)
    self.param_path = 'model.pth'

  def _on_training_start(self):
    state_dict = torch.load(self.param_path, map_location=torch.device(device))

    # Layer 1
    self.model.policy.mlp_extractor.policy_net[0].weight = nn.Parameter(state_dict['l1.weight'])
    self.model.policy.mlp_extractor.policy_net[0].bias = nn.Parameter(state_dict['l1.bias'])

    # Layer 2
    self.model.policy.mlp_extractor.policy_net[2].weight = nn.Parameter(state_dict['l2.weight'])
    self.model.policy.mlp_extractor.policy_net[2].bias = nn.Parameter(state_dict['l2.bias'])
    
    # Layer 3
    self.model.policy.mlp_extractor.policy_net[4].weight = nn.Parameter(state_dict['l3.weight'])
    self.model.policy.mlp_extractor.policy_net[4].bias = nn.Parameter(state_dict['l3.bias'])
    
    # Output layer
    self.model.policy.action_net.weight = nn.Parameter(state_dict['l4.weight'])
    self.model.policy.action_net.bias = nn.Parameter(state_dict['l4.bias'])

    states = []
    inputs = []

    vec_env = self.model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
      action, _states = self.model.predict(obs, deterministic=True)
      obs, rewards, done, info = vec_env.step(action)
      states.append(obs)
      inputs.append(action)

    states = np.squeeze(np.array(states))
    inputs = np.squeeze(np.array(inputs))

    plt.plot(states[:,0], states[:,1])
    plt.show()

    plt.plot(inputs)
    plt.show()

  def _on_step(self):
    return True


callback = CustomCallback()

model_rl.learn(total_timesteps=30000, callback=callback, progress_bar=True)

states = []
inputs = []

vec_env = model_rl.get_env()
obs = vec_env.reset()
for i in range(1000):
  action, _states = model_rl.predict(obs, deterministic=True)
  obs, rewards, done, info = vec_env.step(action)
  states.append(obs)
  inputs.append(action)

states = np.squeeze(np.array(states))
inputs = np.squeeze(np.array(inputs))

plt.plot(states[:,0], states[:,1])
plt.show()

plt.plot(inputs)
plt.show()