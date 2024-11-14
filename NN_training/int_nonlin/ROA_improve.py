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
warnings.filterwarnings("ignore", category=UserWarning, module='gymnasium')

# Device declaration to exploit GPU acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom environment declaration
env = NonLinPendulum_env()

# Policy declaration to impose the structure and activation function type
policy_kwargs = dict(activation_fn=torch.nn.Hardtanh, net_arch=[32, 32, 32])

# Model declaration
model_rl = PPO(
  "MlpPolicy", 
  env, 
  policy_kwargs=policy_kwargs, 
  verbose=1,
  device=device
)

# Custom callback class to handle different events during training:
# - Training start: load the current best weights from the LMI optimization
# - Rollout start: reset the model to the last best weights if the current model is not improving or is infeasible
# - Rollout end: solve the LMI optimization problem to find the new best weights
# - Step: not used yet

# The callback also saves the best model and the current model at the end of each rollout
class CustomCallback(BaseCallback):
  def __init__(self, verbose = 0):
    super().__init__(verbose)
    # Path to import the best weights, LQR is set to True to load the LQR weights
    self.LQR = False
    if self.LQR:
      self.param_path = os.path.abspath(__file__ + '/../LQR/model.pth')
    else:
      self.param_path = os.path.abspath(__file__ + '/../policy.pth')
    # Variables to store the best ROA area during training
    self.best_ROA = 0.0
    # Variable to store current rollout attempt without improvement
    self.n_rollout_no_improvement = 0
    # User imposed limit to the number of rollouts without improvement
    self.n_rollout_limit = 5
    # Flag variable to reset the model to the last best weights
    self.reset = 0
  
  # Function to extract the weights from the model, it is important to clone the data otherwise only a deep copy of the reference is created
  def get_weights(self, model):
    W1 = self.model.policy.mlp_extractor.policy_net[0].weight.data.clone().detach().to(device).numpy()
    b1 = self.model.policy.mlp_extractor.policy_net[0].bias.data.clone().detach().to(device).numpy()
    W2 = self.model.policy.mlp_extractor.policy_net[2].weight.data.clone().detach().to(device).numpy()
    b2 = self.model.policy.mlp_extractor.policy_net[2].bias.data.clone().detach().to(device).numpy()
    W3 = self.model.policy.mlp_extractor.policy_net[4].weight.data.clone().detach().to(device).numpy()
    b3 = self.model.policy.mlp_extractor.policy_net[4].bias.data.clone().detach().to(device).numpy()
    W4 = self.model.policy.action_net.weight.data.clone().detach().to(device).numpy()
    b4 = self.model.policy.action_net.bias.data.clone().detach().to(device).numpy()
    W = [W1, W2, W3, W4]
    b = [b1, b2, b3, b4]
    return W, b
  
  # Callback called on the training start
  def _on_training_start(self):
    
    # Load the best weights
    state_dict = torch.load(self.param_path, map_location=torch.device(device), weights_only=True)    

    # The flag LQR is necessary since the two state_dict have different names for their elements
    if self.LQR: 
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
    else:
      # Layer 1
      self.model.policy.mlp_extractor.policy_net[0].weight = nn.Parameter(state_dict['mlp_extractor.policy_net.0.weight'].clone().detach().requires_grad_(True))
      self.model.policy.mlp_extractor.policy_net[0].bias = nn.Parameter(state_dict['mlp_extractor.policy_net.0.bias'].clone().detach().requires_grad_(True))
      
      # Layer 2
      self.model.policy.mlp_extractor.policy_net[2].weight = nn.Parameter(state_dict['mlp_extractor.policy_net.2.weight'].clone().detach().requires_grad_(True))
      self.model.policy.mlp_extractor.policy_net[2].bias = nn.Parameter(state_dict['mlp_extractor.policy_net.2.bias'].clone().detach().requires_grad_(True))
      
      # Layer 3
      self.model.policy.mlp_extractor.policy_net[4].weight = nn.Parameter(state_dict['mlp_extractor.policy_net.4.weight'].clone().detach().requires_grad_(True))
      self.model.policy.mlp_extractor.policy_net[4].bias = nn.Parameter(state_dict['mlp_extractor.policy_net.4.bias'].clone().detach().requires_grad_(True))
      
      # Output layer
      self.model.policy.action_net.weight = nn.Parameter(state_dict['action_net.weight'].clone().detach().requires_grad_(True))
      self.model.policy.action_net.bias = nn.Parameter(state_dict['action_net.bias'].clone().detach().requires_grad_(True))

    # Reinitialize the optimizer, otherwise the model will keep the previous optimizer state and will not improve
    optim_class = type(self.model.policy.optimizer)
    optim_param = self.model.policy.optimizer.defaults
    self.model.policy.optimizer = optim_class(self.model.policy.parameters(), **optim_param)

    # Just at the beginning of the training the P matrix is the result of the last standalone LMI problem solution
    P = np.load('Test/P.npy')
    self.model.get_env().env_method('set_P', P, P)

  def _on_step(self):
    return True

  # Callback called on the rollout end after the optimization step of sb3 that updates the policy weights. Will be used to solve the LMI on the new weights and update the best weights accordingly.
  # If the policy update is infeasible the reset flag will handle the weights reset at the beginning of the next rollout
  def _on_rollout_end(self):
    
    # Weights and biases extraction
    W, b = self.get_weights(self.model)
    # LMI class initialization
    lmi = LMI(W, b)  
    # LMI problem solution, alpha value empirically set to 0.1 to have a feasible solution that is not too conservative. The alpha value search will not be performed mid-training to save processing time. Meaning that potentially a slightly better ROA solution can be found by performing it after the training.
    P, _, _ = lmi.solve(0.1)
    
    # Following the logic implemented in the LMI P is None for a infeasible solution.
    if P is not None:

      # ROA area calculation
      area = np.pi/np.sqrt(np.linalg.det(P))
      # Saving current policy if feasible
      self.model.save('rollout_model.zip')

      # If the result is the best so far, update the best weights and reset the no improvement counter. Also save the best model.
      if area > self.best_ROA:
        # Reset no improvement counter
        self.n_rollout_no_improvement = 0

        # P matrix update in the environment
        old_P = self.model.get_env().env_method('get_P')
        self.model.get_env().env_method('set_P', P, old_P)

        # Best ROA update
        self.best_ROA = area
        
        # Best weights update
        self.best_W = W
        self.best_b = b

        # Reward update in the environment, the reward is the difference between the current area and the best area, in this case I prefer to give directly the new ROA as a reward to further encourage the policy to find improvements
        self.model.get_env().env_method('set_ROA_reward', self.best_ROA/200.0)

        # Model save
        self.model.save('best_rollout_model.zip')
        print(f'New best model saved, ROA: {self.best_ROA}')
      # In the case of feasible policy update but not optimal
      else:
        # Increment no improvement counter
        self.n_rollout_no_improvement += 1
        print(f"Feasible increment, but not better than best model. Best ROA: {self.best_ROA}")
        print(f'Difference of area: {area - self.best_ROA}')
        print(f"Keeping current P")
        print(f"Current rollout attempt: {self.n_rollout_no_improvement}/{self.n_rollout_limit}")
        
        # Reward update in the environment, the reward is the difference between the current area and the best area so it is a cost in this case
        self.model.get_env().env_method('set_ROA_reward', (area - self.best_ROA)/200.0)

        # If the no improvement counter reaches the limit, reset the model to the last best weights
        if self.n_rollout_no_improvement >= self.n_rollout_limit:
          print(f"Resetting to last best model")
          self.reset = 1
    
    # In the case of infeasible policy update reset the model to the last best weights
    else:
      print(f'Infeasible increment, keeping current P')
      print(f"Resetting to last best model")
      self.reset = 1

  # Callback called on the rollout start, if the reset flag is set the model will be reset to the last best weights
  def _on_rollout_start(self):
    if self.reset:
      self.reset = 0
      self.n_rollout_no_improvement = 0

      self.model.policy.mlp_extractor.policy_net[0].weight = nn.Parameter(torch.tensor(self.best_W[0], requires_grad=True))
      self.model.policy.mlp_extractor.policy_net[0].bias = nn.Parameter(torch.tensor(self.best_b[0], requires_grad=True))
      self.model.policy.mlp_extractor.policy_net[2].weight = nn.Parameter(torch.tensor(self.best_W[1], requires_grad=True))
      self.model.policy.mlp_extractor.policy_net[2].bias = nn.Parameter(torch.tensor(self.best_b[1], requires_grad=True))
      self.model.policy.mlp_extractor.policy_net[4].weight = nn.Parameter(torch.tensor(self.best_W[2], requires_grad=True))
      self.model.policy.mlp_extractor.policy_net[4].bias = nn.Parameter(torch.tensor(self.best_b[2], requires_grad=True))
      self.model.policy.action_net.weight = nn.Parameter(torch.tensor(self.best_W[3], requires_grad=True))
      self.model.policy.action_net.bias = nn.Parameter(torch.tensor(self.best_b[3], requires_grad=True))
    
      # Reinitialize the optimizer, otherwise the model will keep the previous optimizer state and will not improve
      optim_class = type(self.model.policy.optimizer)
      optim_param = self.model.policy.optimizer.defaults
      self.model.policy.optimizer = optim_class(self.model.policy.parameters(), **optim_param)
      self.reset = 0

# Custom evaluation callback to handle the evaluation of the model during training, it automatically saves the best model based on the evaluation score
CustomEvalCallback = EvalCallback(env, best_model_save_path='.', log_path='./logs', eval_freq=1000, deterministic=True, render=False, verbose=0)

# Callback list to handle the different custom callbacks
# callback = CallbackList([CustomCallback(), CustomEvalCallback])
# Callback to not take into account the evaluation callback
callback = CustomCallback()

# Model training
model_rl.learn(total_timesteps=1000000, callback=callback, progress_bar=True)

## Testing the best model after training

# Empty lists to store the states and inputs of the episodes
states = []
inputs = []
episode_state = []
episode_input = []

# Variables to store the total number of episodes and the number of converging episodes
n_tot = 0
n_converging = 0

# Environment reset
vec_env = model_rl.get_env()
obs = vec_env.reset()

# Loop to run the episodes
for i in range(10000):
  action, _states = model_rl.predict(obs, deterministic=True)
  obs, rewards, done, info = vec_env.step(action)
  episode_state.append(obs)
  episode_input.append(action)

  # If the episode is done, store the episode and reset the episode lists
  if done:
    n_tot += 1
    states.append(episode_state)
    inputs.append(episode_input)
    episode_state = []
    episode_input = []

## Result plotting    
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