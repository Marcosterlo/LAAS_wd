from stable_baselines3 import PPO
from NN_training.environments.simple_pendulum_env import Simple_pendulum_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env = Simple_pendulum_env()

policy_kwargs = dict(activation_fn=torch.nn.Hardtanh, net_arch=[32, 32, 32])

model_rl = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

class CustomCallback(BaseCallback):
  def __init__(self, verbose=0):
    super().__init__(verbose)
    self.param_path = 'model.pth'

  def _on_training_start(self):
    state_dict = torch.load(self.param_path, map_location=torch.device(device), weights_only=True)

    # Layer 1
    self.model.policy.mlp_extractor.policy_net[0].weight = nn.Parameter(state_dict['l1.weight'].clone().detach().requires_grad_(True))
    self.model.policy.mlp_extractor.policy_net[0].bias = nn.Parameter(state_dict['l1.bias'].clone().detach().requires_grad_(True))

    # Layer 2
    self.model.policy.mlp_extractor.policy_net[2].weight = nn.Parameter(state_dict['l2.weight'].clone().detach().requires_grad_(True))
    self.model.policy.mlp_extractor.policy_net[2].bias = nn.Parameter(state_dict['l2.bias'].clone().detach().requires_grad_(True))
    
    # Layer 3
    self.model.policy.mlp_extractor.policy_net[4].weight = nn.Parameter(state_dict['l3.weight'].clone().detach().requires_grad_(True))
    self.model.policy.mlp_extractor.policy_net[4].bias = nn.Parameter(state_dict['l3.bias'].clone().detach().requires_grad_(True))
    
    # Output layer
    self.model.policy.action_net.weight = nn.Parameter(state_dict['l4.weight'].clone().detach().requires_grad_(True))
    self.model.policy.action_net.bias = nn.Parameter(state_dict['l4.bias'].clone().detach().requires_grad_(True))

  def _on_step(self):
    return True
  
  def _on_rollout_end(self):
    pass

CustomEvalCallback = EvalCallback(env, best_model_save_path='.', log_path='./logs', eval_freq=1000, deterministic=True, render=False, verbose=0)

callback = CallbackList([CustomCallback(), CustomEvalCallback])

model_rl.learn(total_timesteps=30000, callback=callback, progress_bar=True)

best_model = PPO.load('best_model.zip', env=env)

states = []
inputs = []
episode_state = []
episode_input = []

vec_env = best_model.get_env()
obs = vec_env.reset()
for i in range(10000):
  action, _states = best_model.predict(obs, deterministic=True)
  obs, rewards, done, info = vec_env.step(action)
  episode_state.append(obs)
  episode_input.append(action)
  if done:
    states.append(episode_state)
    inputs.append(episode_input)
    episode_state = []
    episode_input = []

for episode in states:
  if len(episode) > 1:
    episode = np.squeeze(np.array(episode))
    plt.plot(episode[:,0], episode[:,1])
plt.show()

for episode in inputs:
  episode = np.squeeze(np.array(episode))
  plt.plot(episode)
plt.show()