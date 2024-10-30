from systems_and_LMI.systems.NonLinearPendulum import NonLinPendulum
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class Pendulum_env(gym.Env):
  
  def __init__(self):
    super(Pendulum_env, self).__init__()

    self.system = NonLinPendulum()
    
    self.g = self.system.g
    self.m = self.system.m
    self.l = self.system.l
    self.mu = self.system.mu
    self.dt = self.system.dt
    self.max_torque = self.system.max_torque
    self.max_speed = self.system.max_speed

    self.A = self.system.A
    self.B = self.system.B
    self.C = self.system.C
    self.D = self.system.D

    self.nx = self.system.nx