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

        self.state = None

        self.A = np.array([[1,      self.dt],
                  [self.g/self.l*self.dt, 1 - self.mu/(self.m*self.l**2)*self.dt]])

        self.B = np.array([[0], [self.dt/(self.m*self.l**2)]])

        self.K = np.array([-0.1, 0]).reshape(1,2)

        self.time = 0

        self.last_u = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.nu,))
        self.xmax = np.array([np.pi, self.max_speed])
        self.observation_space = spaces.Box(low=-self.xmax, high=self.xmax)

    def step(self, action):

        th, thdot = self.state
        
        A = self.A
        B = self.B
        K = self.K

        u = np.clip(action, -1, 1) * self.max_torque

        ## Cost function with differential u
        reward = np.exp(-th**2) + 0.1*np.exp(-thdot**2) + 0.01*np.exp(-u**2)
        reward = -th**2 - 0.1*thdot**2 - 0.01*u**2 + 1
        
        self.state = np.squeeze((A + B @ K) @ self.state.reshape(2,1) + B @ u.reshape(1,1)).astype(np.float32)


        terminated = False
        if self.time > 200 or not self.observation_space.contains(self.state):
            terminated = True
        
        self.time += 1

        return self.get_obs(), float(reward), terminated, terminated, {}

    def reset(self, seed=None):
        xlim = np.pi/2
        vlim = 2
        self.state = np.array([np.random.uniform(-xlim, xlim), np.random.uniform(-vlim, vlim)]).astype(np.float32)
        self.time = 0
        return (self.get_obs(), {})
    
    def get_obs(self):
        return self.state
    
    def render(self):
        print(f"Pendulum state: theta={self.state[0]:.2f}, theta_dot={self.state[1]:.2f}")