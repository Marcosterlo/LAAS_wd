import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from systems_and_LMI.systems.NonLinPendulum_no_int import NonLinPendulum_no_int

class Simple_pendulum_env(gym.Env):
  
  def __init__(self):
    super(Simple_pendulum_env, self).__init__()

    # Initialize the system to get the parameters
    self.system = NonLinPendulum_no_int()
    self.max_speed = self.system.max_speed
    self.max_torque = self.system.max_torque
    self.dt = self.system.dt
    self.g = self.system.g
    self.m = self.system.m
    self.l = self.system.l

    # Action space definition
    self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    # Observation space definition
    self.lim_state = np.array([np.pi/2, self.max_speed], dtype=np.float32)
    self.observation_space = spaces.Box(low=-self.lim_state, high=self.lim_state, shape=(2,), dtype=np.float32)

    # Time variable
    self.time = 0
  
  def step(self, action):

    # Theta and theta dot extraction
    th, thdot = self.state[0]
    
    # Action clipping and de-normalization
    action = np.clip(action, -1.0, 1.0) * self.max_torque

    # Theta normalization
    th = (th + np.pi) % (2*np.pi) - np.pi

    # Cost computation
    cost = th**2 + 0.1*thdot**2 + 0.001*(action**2) - 1

    # Dynamics update
    new_state = self.system.A @ self.state.reshape(2,) + (self.system.B * action).reshape(2,) + (self.system.C * (np.sin(th) - th)).reshape(2,)

    # State update
    self.state = np.array([new_state.astype(np.float32)])

    truncated = False
    terminated = False

    # End of episode defintion
    if self.time >= 200 - 1:
      truncated = True

    # Early termination definition    
    if self.state not in self.observation_space:
      terminated = True

    # Time update
    self.time += 1

    return self.get_obs(), -float(cost), terminated, truncated, {}

  def reset(self, seed=None):
    # Random initial state
    self.state = np.array([np.random.uniform(low=-self.lim_state, high=self.lim_state)])
    # Reset time
    self.time = 0
    return (self.get_obs(), {})
  
  def get_obs(self):
    th, thdot = self.state[0]
    return np.array([[th, thdot]], dtype=np.float32)
  
  def render(self):
    th, thdot = self.get_obs()
    print(f"Theta: {th:.2f}, Theta dot: {thdot:.2f}")
  
if __name__ == "__main__":
  env = Simple_pendulum_env()
  check_env(env)