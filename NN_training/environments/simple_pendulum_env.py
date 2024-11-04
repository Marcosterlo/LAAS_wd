import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
import torch
from systems_and_LMI.systems.NonLinPendulum_no_int import NonLinPendulum_no_int

class Simple_pendulum_env(gym.Env):
  
  def __init__(self):
    super(Simple_pendulum_env, self).__init__()

    self.system = NonLinPendulum_no_int()

    self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

    self.time = 0
    self.state = None
  
  def step(self, action):
    
    action = np.clip(action, -1.0, 1.0)*self.system.max_torque
    
    return self.state, reward, done, {}