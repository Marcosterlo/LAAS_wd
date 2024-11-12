import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from systems_and_LMI.systems.NonLinearPendulum import NonLinPendulum

class NonLinPendulum_env(gym.Env):
  
  def __init__(self):
    super(NonLinPendulum_env, self).__init__()
    
    # Initialize the system to get the parameters
    # Reference initialization
    self.ref_bound = 0.5
    self.ref = 0.0
    self.system = NonLinPendulum(self.ref)
    self.max_speed = self.system.max_speed
    self.max_torque = self.system.max_torque
    self.dt = self.system.dt
    self.g = self.system.g
    self.m = self.system.m
    self.l = self.system.l

    # Dynamics matrices
    self.A = self.system.A
    self.B = self.system.B
    self.C = self.system.C
    self.D = self.system.D
    
    # Action space definition
    self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    
    # Observation space definition
    self.lim_state = np.array([np.pi, self.max_speed, np.inf], dtype=np.float32)
    self.observation_space = spaces.Box(low=-self.lim_state, high=self.lim_state, shape=(3,), dtype=np.float32)
    
    # Time variable
    self.time = 0

    # Empty state initialization
    self.state = None
    
  def step(self, action):

    th, thdot, eta = self.state
    
    action = np.clip(action, -1.0, 1.0) * self.max_torque

    th = self.state[0] 
    th = (th + np.pi) % (2*np.pi) - np.pi

    cost = (th**2 + 0.1*thdot**2 + + 0.001*(eta**2) + 0.001*(action**2) - 1)[0]

    state = np.squeeze(self.state)

    new_state = self.A @ state + (self.B * action).reshape(3,) + (self.C * (np.sin(th) - th)).reshape(3,) + (self.D * self.ref).reshape(3,)

    self.state = np.squeeze(np.array([new_state.astype(np.float32)]))

    truncated = False
    terminated = False
    
    if self.time >= 200 - 1:
      truncated = True
      
    if not self.observation_space.contains(self.state):
      terminated = True
      
    self.time += 1
    
    return self.get_obs(), -float(cost), terminated, truncated, {}
  
  def reset(self, seed=None):
    th = np.float32(np.random.uniform(low=-self.lim_state[0], high=self.lim_state[0]))
    thdot = np.float32(np.random.uniform(low=-self.lim_state[1], high=self.lim_state[1]))
    eta = np.float32(0.0)
    self.state = np.squeeze(np.array([th, thdot, eta]))
    self.time = 0
    self.ref = np.random.uniform(-self.ref_bound, self.ref_bound)
    return (self.get_obs(), {})
  
  def get_obs(self):
    return self.state

if __name__ == "__main__":
  env = NonLinPendulum_env()
  check_env(env)