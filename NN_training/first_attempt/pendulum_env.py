import numpy as np
import gymnasium as gym
from gymnasium import spaces

class Pendulum_env(gym.Env):

    def __init__(self):
        super(Pendulum_env, self).__init__()

        self.g = 9.81 # grav coeff
        self.m = 0.15 # mass
        self.l = 0.5 # lenth
        self.mu = 0.05 # fric coeff
        self.dt = 0.02 # sampling period

        self.max_torque = 10
        self.max_speed = 8.0

        self.nx = 2
        self.nu = 1

        self.time = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.nu,))
        self.xmax = np.array([np.pi/2, self.max_speed])
        self.observation_space = spaces.Box(low=-self.xmax, high=self.xmax)

        self.last_u = 0

    def step(self, action):

        th, thdot = self.state

        g = self.g
        m = self.m
        l = self.l
        mu = self.mu
        dt = self.dt

        u = np.clip(action * self.max_torque, -1, 1)

        ## Cost function definition
        costs = float(th**2 + 0.01*thdot**2 + 0.01*(u**2) + 0.001*(u - self.last_u)**2 - 1)
        

        # Non linear behavior
        # theta_ddot = g/l*np.sin(th) - mu/(m*l**2)*thdot + 1/(m*l**2)*u

        # Linear behavior
        theta_ddot = g/l*th - mu/(m*l**2)*thdot + 1/(m*l**2)*u

        thdot += theta_ddot * dt
        th += thdot * dt

        self.state = np.squeeze(np.array([th, thdot]).astype(np.float32))

        terminated = False
        if self.time > 200 or not self.observation_space.contains(self.state):
            terminated = True
        
        self.time += 1

        return self.get_obs(), -costs, terminated, terminated, {}

    def reset(self, seed=None):
        xlim = self.xmax[0]
        vlim = self.xmax[1]
        self.state = np.array([np.random.uniform(-xlim, xlim), np.random.uniform(-vlim, vlim)]).astype(np.float32)
        self.time = 0
        return (self.get_obs(), {})
    
    def get_obs(self):
        return self.state
    
    def render(self):
        print(f"Pendulum state: theta={self.state[0]:.2f}, theta_dot={self.state[1]:.2f}")