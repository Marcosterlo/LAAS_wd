import numpy as np
import gymnasium as gym
from gymnasium import spaces

class Integrator_env(gym.Env):

    def __init__(self):
        super(Integrator_env, self).__init__()

        # state in the form x, eta
        self.state = None

        self.nx = 
        self.nu = 

        self.constant_reference = 0.2
        self.max_input = 5

        self.A = 
        self.B = 
        self.C = 

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.nu,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf)
    
    def step(self, action):

        x, eta = self.state

        A = self.A
        B = self.B
        C = self.C

        y = C @ x

        u = np.clip(action, -1, 1) * self.max_input

        xplus = A @ x + phi1(y) + B @ u
        etaplus = eta - (y - self.constant_reference)

        self.state = np.array([[xplus], [etaplus]])

        terminated = False
        if self.time > 200 or not self.observation_space.contains(self.state):
            terminated = True

        self.time += 1

        return self.get_obs(), float(reward), terminated, terminated, {}
    
    def reset(self, seed=None):
        x_lim = 0.5
        eta_lim = 0.5
        newx = np.random.uniform(-x_lim, x_lim)
        neweta = np.random.uniform(-eta_lim, eta_lim)
        self.state = np.array([[newx], [neweta]])
        self.time = 0

        return (self.get_obs(), {})
    
    def get_obs(self):
        x = self.state[0]
        y = self.C @ x
        eta = self.state[1]
        return np.array([[y], [eta]])
    
    def render(self):
        print(f"State x: {}")
