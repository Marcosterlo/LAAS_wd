import numpy as np
import os
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
import systems_and_LMI.LMI.no_int_3l.main as lmi

class Nn_3l_env(gym.Env):
  
  def __init__(self):
    super(Nn_3l_env, self).__init__()
    
    W1_name = os.path.abspath(__file__ + "/../../../systems_and_LMI/systems/simple_weights/l1.weight.csv")
    W2_name = os.path.abspath(__file__ + "/../../../systems_and_LMI/systems/simple_weights/l2.weight.csv")
    W3_name = os.path.abspath(__file__ + "/../../../systems_and_LMI/systems/simple_weights/l3.weight.csv")
    W4_name = os.path.abspath(__file__ + "/../../../systems_and_LMI/systems/simple_weights/l4.weight.csv")

    W1 = np.loadtxt(W1_name, delimiter=',')
    W2 = np.loadtxt(W2_name, delimiter=',')
    W3 = np.loadtxt(W3_name, delimiter=',')
    W4 = np.loadtxt(W4_name, delimiter=',')
    W4 = W4.reshape((1, len(W4)))

    self.W = [W1, W2, W3, W4]

    b1_name = os.path.abspath(__file__ + "/../../../systems_and_LMI/systems/simple_weights/l1.bias.csv")
    b2_name = os.path.abspath(__file__ + "/../../../systems_and_LMI/systems/simple_weights/l2.bias.csv")
    b3_name = os.path.abspath(__file__ + "/../../../systems_and_LMI/systems/simple_weights/l3.bias.csv")
    b4_name = os.path.abspath(__file__ + "/../../../systems_and_LMI/systems/simple_weights/l4.bias.csv")
    
    b1 = np.loadtxt(b1_name, delimiter=',')
    b2 = np.loadtxt(b2_name, delimiter=',')
    b3 = np.loadtxt(b3_name, delimiter=',')
    b4 = np.loadtxt(b4_name, delimiter=',')

    self.b = [b1, b2, b3, b4]
      
    # Normalized action space definition
    self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.get_num_params(),), dtype=np.float32)
    
    # Maximum values for state and observation space definition
    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.get_num_params(),), dtype=np.float32)

    # Variables to store the last best feasible set of weights and biases
    self.last_W = self.W
    self.last_b = self.b

    # Variable to store the best eigenvalue found, or best ROA
    self.best_ROA = 1e-10

    # Variable to store length of the episode
    self.episode_length = 10

    # Variable to store current episode step
    self.current_step = 0

    # Flag variable to update current weights and biases
    self.update_request = False

    print(f'Setup values')
    self.reward(self.W, self.b)

  # This method gives the size of the action and the observation space being the number of parameters of the neural network
  def get_num_params(self):
    num_params = 0
    for weight in self.W:
      num_params += weight.size
    for bias in self.b:
      num_params += bias.size
    return num_params
  
  def step(self, action):
    
    # Importation of new weights and biases
    W, b = self.get_weights(action)

    terminated = False

    reward = self.reward(W, b)

    if self.current_step >= self.episode_length - 1:
      terminated = True

    self.current_step += 1
    
    return self.get_obs(), reward, terminated, terminated, {}
  
  # The function get_weigth will return the new weights and biases after the increment action is applied
  def get_weights(self, increment_action):
    offset = 0
    new_W = []
    new_b = []
    for weight in self.W:
      weight_size = weight.size
      weight += 1e-3 * increment_action[offset:offset + weight_size].reshape(weight.shape)
      offset += weight_size
      new_W.append(weight)
    
    for bias in self.b:
      bias_size = bias.size
      bias += 1e-3 * increment_action[offset:offset + bias_size].reshape(bias.shape)
      offset += bias_size
      new_b.append(bias)

    return new_W, new_b

  # Function that handles the reward function. It will execute the LMI and compute the reward accordingly
  def reward(self, W, b):
    lmi_obj = lmi.LMI_3l_no_int(W, b)
    #alpha = lmi_obj.search_alpha(0.1, 0, 1e-2, verbose=True)
    P, _, _ = lmi_obj.solve(0.02)
    if P is None:
      return float(0.0)
    else:
      area = np.pi / np.sqrt(np.linalg.det(P))
      if area > self.best_ROA:
        self.best_ROA = area
        print(f"Found better ROA: current area: {self.best_ROA:.2f}, current max P eigenvalue: {np.max(np.linalg.eigvals(P)):.2f}")
        self.last_W = W
        self.last_b = b
        self.update_request = True
        print("Updating weights and biases")

      reward = float(np.log(1 + area))
      return float(reward)
  
  # The reset method puts the last best feasible set of weight and biases
  def reset(self, seed=None):
    self.current_step = 0
    if self.update_request:
      self.update_request = False
      self.W = self.last_W
      self.b = self.last_b
    return (self.get_obs(), {})
  
  # Function to extract the observation being the ordered list of element of all the weights and biases
  def get_obs(self):
    obs = []
    for w in self.W:
      obs.append(w.flatten().astype(np.float32))
    for b in self.b:
      obs.append(b.flatten().astype(np.float32))
    return np.concatenate(obs)
    
# Main execution to check if the environment is correctly defined following the OpenAI Gym standards API
if __name__ == "__main__":
  env = Nn_3l_env()
  check_env(env, warn=True)