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

    # P matrix for lypaunov function used for cost
    self.P = np.eye(self.system.nx)
    self.old_P = np.eye(self.system.nx)*0
    
    # Action space definition
    self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    
    # Observation space definition
    self.lim_state = np.array([np.pi, self.max_speed, np.inf], dtype=np.float32)
    self.observation_space = spaces.Box(low=-self.lim_state, high=self.lim_state, shape=(3,), dtype=np.float32)
    
    # Time variable
    self.time = 0
    # End of episode
    self.episode_end = 200

    # Empty state initialization
    self.state = None

    self.ROA_reward = 0.0

  # Setter methods
  def set_P(self, P, old_P):
    self.P = P
    self.old_P = old_P
  
  def set_ROA_reward(self, ROA_reward):
    self.ROA_reward = ROA_reward
  
  # Getter methods
  def get_P(self):
    return self.P

  def step(self, action):

    th, thdot, eta = self.state
    
    action = np.clip(action, -1.0, 1.0) * self.max_torque

    th = self.state[0] 
    th = (th + np.pi) % (2*np.pi) - np.pi

    state = np.squeeze(self.state)

    new_state = self.A @ state + (self.B * action).reshape(3,) + (self.C * (np.sin(th) - th)).reshape(3,) + (self.D * self.ref).reshape(3,)

    self.state = np.squeeze(np.array([new_state.astype(np.float32)]))

    LMI_cost = True

    if LMI_cost:
      xstar = np.squeeze(self.system.xstar)
      cost = (new_state - xstar).T @ self.P @ (new_state - xstar) - (state - xstar).T @ self.P @ (state - xstar)
    else: 
      state_cost = (th**2 + 0.1*thdot**2 +  0.001*(eta**2) + 0.001*(action**2))[0]
      stay_alive_reward = 1.0
      
      initial_state_weight = 1.0 
      final_state_weight = 1.0
      state_weight = (final_state_weight - initial_state_weight) /self.episode_end * self.time + initial_state_weight

      initial_reward_weight = 1.0
      final_reward_weight = 1.0
      stay_alive_weight = (final_reward_weight - initial_reward_weight) /self.episode_end * self.time + initial_reward_weight

      cost = state_cost * state_weight - stay_alive_reward * stay_alive_weight
      cost = state_cost - stay_alive_reward

    truncated = False
    terminated = False
    
    if self.time >= 200 - 1:
      truncated = True
      
    state_to_check = (1/self.episode_end * self.time + 1) * self.state
    if not self.observation_space.contains(state_to_check):
      terminated = True
      
    self.time += 1

    return self.get_obs(), -float(cost) + self.ROA_reward, terminated, truncated, {}
  
  def reset(self, seed=None):
    th = np.float32(np.random.uniform(low=-self.lim_state[0], high=self.lim_state[0]))
    thdot = np.float32(np.random.uniform(low=-self.lim_state[1], high=self.lim_state[1]))
    eta = np.float32(0.0)
    self.state = np.squeeze(np.array([th, thdot, eta]))
    # if self.time != 0:
      # print(f'Current episode length: {self.time}')
      # print(f"Diff in P: {np.linalg.norm(self.P - self.old_P)}")
    self.time = 0
    self.ref = np.random.uniform(-self.ref_bound, self.ref_bound)
    self.system = NonLinPendulum(self.ref)
    return (self.get_obs(), {})
  
  def get_obs(self):
    return self.state

if __name__ == "__main__":
  env = NonLinPendulum_env()
  check_env(env)