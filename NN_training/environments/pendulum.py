from systems_and_LMI.systems.NonLinPendulum_no_int_train import NonLinPendulum_no_int_train
import numpy as np
import os
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
import torch

class Pendulum_env(gym.Env):
  
  def __init__(self, model):
    super(Pendulum_env, self).__init__()


    W1_name = os.path.abspath(__file__ + "/../../../systems_and_LMI/systems/simple_weights/l1.weight.csv")
    W2_name = os.path.abspath(__file__ + "/../../../systems_and_LMI/systems/simple_weights/l2.weight.csv")
    W3_name = os.path.abspath(__file__ + "/../../../systems_and_LMI/systems/simple_weights/l3.weight.csv")
    W4_name = os.path.abspath(__file__ + "/../../../systems_and_LMI/systems/simple_weights/l4.weight.csv")

    W1 = np.loadtxt(W1_name, delimiter=',')
    W2 = np.loadtxt(W2_name, delimiter=',')
    W3 = np.loadtxt(W3_name, delimiter=',')
    W4 = np.loadtxt(W4_name, delimiter=',')
    W4 = W4.reshape((1, len(W4)))

    W = [W1, W2, W3, W4]

    b1_name = os.path.abspath(__file__ + "/../../../systems_and_LMI/systems/simple_weights/l1.bias.csv")
    b2_name = os.path.abspath(__file__ + "/../../../systems_and_LMI/systems/simple_weights/l2.bias.csv")
    b3_name = os.path.abspath(__file__ + "/../../../systems_and_LMI/systems/simple_weights/l3.bias.csv")
    b4_name = os.path.abspath(__file__ + "/../../../systems_and_LMI/systems/simple_weights/l4.bias.csv")
    
    b1 = np.loadtxt(b1_name, delimiter=',')
    b2 = np.loadtxt(b2_name, delimiter=',')
    b3 = np.loadtxt(b3_name, delimiter=',')
    b4 = np.loadtxt(b4_name, delimiter=',')
    
    b = [b1, b2, b3, b4] 

    self.system = NonLinPendulum_no_int_train(W, b)

    # Initialization of model that will be used to extract parameters and execute the LMI
    self.model = model
    
    # Normalized action space definition
    self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.get_num_params(),), dtype=np.float32)
    
    # Maximum values for state and observation space definition
    xmax = np.array([np.pi/2, self.system.max_speed])
    self.observation_space = spaces.Box(low=-xmax, high=xmax, shape=(self.system.nx,))

    # Variable to track episode length
    self.time = 0
  
  def get_num_params(self):
    return sum(p.numel() for p in self.model.parameters())
  
  # Necessary function to execute a step of the system. It won't execute directly the step function of the system since it refers to its own trained NN controller
  def step(self, action):

    W, b = self.get_weights(action)

    self.system = NonLinPendulum_no_int_train(W, b)
    
    # Extraction of theta to compute the non linear term
    theta, _ = self.system.state

    # Dynamics of the system execution
    self.system.state.step()

    # Termination condition
    terminated = False
    if self.time > 300 or not self.observation_space.contains(self.system.state):
      terminated = True
    
    # Time progression
    self.time += 1

    return self.get_obs(), float(self.reward()), terminated, terminated, {}
  
  def get_weights(self, weights):
    W = [] 
    b = []
    offset = 0
    for param in self.model.parameters():
      param_length = param.numel()
      data = weights[offset:offset + param_length]
      W.append(data)
      b.append(data)
      offset += param_length
    return W, b    

  # Function that handles the reward function. It will execute the LMI and compute the reward accordingly
  def reward(self):
    return 1.0
  
  # Function to reset the environment in a new intial state potentially included in the ROA
  def reset(self, seed=None):
    theta_lim = 60 * np.pi / 180
    vtheta_lim = self.system.max_speed * 0.8
    theta = np.random.uniform(-theta_lim, theta_lim)
    vtheta = np.random.uniform(-vtheta_lim, vtheta_lim)
    self.system.state = np.array([[theta], [vtheta]])
    self.time = 0
    return (self.get_obs(), {})
  
  # Function to extract the observation from the system state
  def get_obs(self):
    theta = self.system.state[0]
    vtheta = self.system.state[1]
    return np.squeeze(np.array([[theta], [vtheta]]).astype(np.float32))
  
  # Function to print the current state of the system
  def render(self):
    state = self.get_obs()
    print(f"State: theta = {state[0] * 180 / np.pi:.2f}, vtheta = {state[1]:.2f}")
    
# Main execution to check if the environment is correctly defined following the OpenAI Gym standards API
if __name__ == "__main__":
  from NN_training.models.NeuralNet_simple import NeuralNet
  model = NeuralNet()
  env = Pendulum_env(model)
  check_env(env, warn=True)