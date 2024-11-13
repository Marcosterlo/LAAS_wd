from stable_baselines3 import PPO
from NN_training.environments.pendulum_integrator_env import NonLinPendulum_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from systems_and_LMI.LMI.int_3l.main import LMI_3l_int as LMI
import warnings
import os

# User warnings filter
warnings.filterwarnings("ignore", category=UserWarning, module='stable_baselines3')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env = NonLinPendulum_env()

policy_kwargs = dict(activation_fn=torch.nn.Hardtanh, net_arch=[32, 32, 32])

model_rl = PPO(
  "MlpPolicy", 
  env, 
  policy_kwargs=policy_kwargs, 
  learning_rate=1,
  n_steps=2048,
  n_epochs=10,
  batch_size=64,
  clip_range=0.2,
  verbose=1
)

class CustomCallback(BaseCallback):
  def __init__(self, verbose = 0):
    super().__init__(verbose)
    self.param_path = os.path.abspath(__file__ + '/../pole_placement/model.pth')
    self.best_ROA = 0.0
    self.prev_w = None
  
  def get_weights(self, model):
    # for name, param in model.policy.named_parameters():
    #   if 'policy' in name or 'action' in name:
    #     if 'weight' in name:
    #       if '0' in name:
    #         W1 = param.clone().detach().numpy()
    #       elif '2' in name:
    #         W2 = param.clone().detach().numpy()
    #       elif '4' in name:
    #         W3 = param.clone().detach().numpy()
    #       elif 'action' in name:
    #         W4 = param.clone().detach().numpy()
    #     if 'bias' in name:
    #       if '0' in name:
    #         b1 = param.clone().detach().numpy()
    #       elif '2' in name:
    #         b2 = param.clone().detach().numpy()
    #       elif '4' in name:
    #         b3 = param.clone().detach().numpy()
    #       elif 'action' in name:
    #         b4 = param.clone().detach().numpy()

    W1 = self.model.policy.mlp_extractor.policy_net[0].weight.data.clone().detach()
    b1 = self.model.policy.mlp_extractor.policy_net[0].bias.data.clone().detach().numpy()
    W2 = self.model.policy.mlp_extractor.policy_net[2].weight.data.clone().detach().numpy()
    b2 = self.model.policy.mlp_extractor.policy_net[2].bias.data.clone().detach().numpy()
    W3 = self.model.policy.mlp_extractor.policy_net[4].weight.data.clone().detach().numpy()
    b3 = self.model.policy.mlp_extractor.policy_net[4].bias.data.clone().detach().numpy()
    W4 = self.model.policy.action_net.weight.data.clone().detach().numpy()
    b4 = self.model.policy.action_net.bias.data.clone().detach().numpy()
    W = [W1, W2, W3, W4]
    b = [b1, b2, b3, b4]
    return W, b
  
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

    P = np.load('Test/P.npy')
    self.model.get_env().set_attr('P', P)

  def _on_step(self):
    return True
  
  def _on_rollout_end(self):
    # w = self.model.policy.action_net.weight.data.clone()
    # if self.prev_w is not None:
    #   diff = w - self.prev_w
    #   max_diff = torch.max(torch.abs(diff)).item()
    #   print(f'Max_diff: {max_diff}')
    # self.prev_w = w.clone()

    W, b = self.get_weights(self.model)
    lmi = LMI(W, b)  
    P, _, _ = lmi.solve(0.1)
    if P is not None:
      area = np.pi/np.sqrt(np.linalg.det(P))
      old_P = self.model.get_env().get_attr('P')
      self.model.get_env().set_attr('old_P', old_P)
      self.model.get_env().set_attr('P', P)
      self.model.save('rollout_model.zip')
      if area > self.best_ROA:
        self.best_ROA = area
        self.best_W = W
        self.best_b = b
        self.model.save('best_rollout_model.zip')
        print('New best model saved')
      else:
        print(f"Feasible increment, but not better than best model. Best ROA: {self.best_ROA}")
        print(f'Difference of area: {area - self.best_ROA}')
    else:
      print(f'Infeasible increment, keeping current P')

    # self.model.policy.mlp_extractor.policy_net[0].weight = nn.Parameter(torch.tensor(self.best_W[0], requires_grad=True))
    # self.model.policy.mlp_extractor.policy_net[0].bias = nn.Parameter(torch.tensor(self.best_b[0], requires_grad=True))
    # self.model.policy.mlp_extractor.policy_net[2].weight = nn.Parameter(torch.tensor(self.best_W[1], requires_grad=True))
    # self.model.policy.mlp_extractor.policy_net[2].bias = nn.Parameter(torch.tensor(self.best_b[1], requires_grad=True))
    # self.model.policy.mlp_extractor.policy_net[4].weight = nn.Parameter(torch.tensor(self.best_W[2], requires_grad=True))
    # self.model.policy.mlp_extractor.policy_net[4].bias = nn.Parameter(torch.tensor(self.best_b[2], requires_grad=True))
    # self.model.policy.action_net.weight = nn.Parameter(torch.tensor(self.best_W[3], requires_grad=True))
    # self.model.policy.action_net.bias = nn.Parameter(torch.tensor(self.best_b[3], requires_grad=True))
  
CustomEvalCallback = EvalCallback(env, best_model_save_path='.', log_path='./logs', eval_freq=1000, deterministic=True, render=False, verbose=0)

callback = CallbackList([CustomCallback(), CustomEvalCallback])

model_rl.learn(total_timesteps=300000, callback=callback, progress_bar=True)

states = []
inputs = []
episode_state = []
episode_input = []

n_tot = 0
n_converging = 0

vec_env = model_rl.get_env()
obs = vec_env.reset()
for i in range(10000):
  action, _states = model_rl.predict(obs, deterministic=True)
  obs, rewards, done, info = vec_env.step(action)
  episode_state.append(obs)
  episode_input.append(action)
  if done:
    n_tot += 1
    states.append(episode_state)
    inputs.append(episode_input)
    episode_state = []
    episode_input = []
    
for episode in states:
  if len(episode) > 50:
    n_converging += 1
    episode = np.squeeze(np.array(episode))
    plt.plot(episode[:,0], episode[:,1])
plt.show()

for episode in inputs:
  if len(episode) > 50:
    episode = np.squeeze(np.array(episode))
    plt.plot(episode)
plt.show()

print(f'Converging episodes: {n_converging}/{n_tot}')