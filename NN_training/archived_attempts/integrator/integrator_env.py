import numpy as np
import gymnasium as gym
from gymnasium import spaces

class Integrator_env(gym.Env):

    def __init__(self):
        super(Integrator_env, self).__init__()

        # state in the form x, eta
        self.state = None

        self.g = 9.81
        self.m = 0.15
        self.l = 0.5
        self.mu = 0.05
        self.dt = 0.02
        self.max_torque = 5
        self.max_speed = 8.0

        self.constant_reference = 0

        self.nx = 3
        self.nu = 1

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.nu,))
        xmax = np.array([np.pi/2, self.max_speed, np.inf])
        self.observation_space = spaces.Box(low=-xmax, high=xmax, shape=(self.nx,))

        self.time = 0
        self.last_eta = 1e3
    
    def step(self, action):

        x, dx, eta = self.state
        g = self.g
        m = self.m
        l = self.l
        mu = self.mu
        dt = self.dt

        y = x - self.constant_reference

        u = np.squeeze(np.clip(action, -1, 1) * self.max_torque)

        # reward =  - (x)**2 - 0.1*dx**2 - (eta - self.last_eta)**2 - 0.01*u**2 + 1
        
        W_y = 1
        W_dx = 0.01
        W_u = 0.001 
        W_eta = 0.01
        W_eta_y = 1

        vec = np.array([y, dx, eta, u])
        W = np.array([
            [W_y,         0,      0.5*W_eta_y,    0],
            [0,           W_dx,   0.005,              0],
            [0.5*W_eta_y, 0.005,      W_eta,          0],
            [0,           0,      0,              W_u]
        ])
        reward = -vec.T @ W @ vec + 1

        dxplus = dx + (g/l*np.sin(x) - mu/(m*l**2)*dx + 1/(m*l**2)*u) * dt
        xplus = x + dx * dt
        etaplus = eta + y
        self.last_eta = etaplus

        self.state = np.array([xplus, dxplus, etaplus]).astype(np.float32)

        terminated = False
        if self.time > 500 or not self.observation_space.contains(self.state):
            terminated = True

        self.time += 1

        return self.get_obs(), float(reward), terminated, terminated, {}
    
    def reset(self, seed=None):
        x_lim = 60 * np.pi / 180
        dx_lim = self.max_speed
        newx = np.random.uniform(-x_lim, x_lim)
        newdx = np.random.uniform(-dx_lim, dx_lim)
        neweta = 0
        self.state = np.array([newx, newdx, neweta]).astype(np.float32)
        self.time = 0

        return (self.get_obs(), {})
    
    def get_obs(self):
        x = self.state[0]
        dx = self.state[1]
        eta = self.state[2]
        return np.array([x, dx, eta]).astype(np.float32)
    
    def render(self):
        state = self.get_obs()
        x = state[0]
        dx = state[1]
        eta = state[2]
        print(f"State x: {x:.2f}, dx: {dx:.2f}, eta: {eta:.2f}")
