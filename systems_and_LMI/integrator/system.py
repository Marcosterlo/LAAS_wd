import numpy as np
import os
from torch import nn
import torch
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

class Integrator():

    def __init__(self):

        self.state = None
        self.g = 9.81
        self.m = 0.5
        self.l = 0.5
        self.mu = 0.05
        self.dt = 0.02
        self.max_torque = 5
        self.max_speed = 8.0
        self.constant_reference = 0
        self.nx = 3
        self.nu = 1

        self.A = np.array([
            [1, self.dt, 0],
            [self.g*self.dt/self.l, 1-self.mu*self.dt/(self.m*self.l**2), 0],
            [1, 0, 1]
        ])
        self.B = np.array([
            [0],
            [self.dt/(self.m*self.l**2)],
            [0]
        ])
        self.Mq = np.array([0, self.g*self.dt/self.l, 0])

        W1_name = os.path.abspath(__file__ + "/../weights/W1.csv")
        W2_name = os.path.abspath(__file__ + "/../weights/W2.csv")
        W3_name = os.path.abspath(__file__ + "/../weights/W3.csv")
        W4_name = os.path.abspath(__file__ + "/../weights/W4.csv")

        W1 = np.loadtxt(W1_name, delimiter=',')
        W2 = np.loadtxt(W2_name, delimiter=',')
        W3 = np.loadtxt(W3_name, delimiter=',')
        W4 = np.loadtxt(W4_name, delimiter=',')
        W4 = W4.reshape(self.nu, len(W4))

        self.W = [W1, W2, W3, W4]

        self.nphi = W1.shape[0] + W2.shape[0] + W3.shape[0]

        b1_name = os.path.abspath(__file__ + "/../weights/b1.csv")
        b2_name = os.path.abspath(__file__ + "/../weights/b2.csv")
        b3_name = os.path.abspath(__file__ + "/../weights/b3.csv")
        b4_name = os.path.abspath(__file__ + "/../weights/b4.csv")

        b1 = np.loadtxt(b1_name, delimiter=',')
        b2 = np.loadtxt(b2_name, delimiter=',')
        b3 = np.loadtxt(b3_name, delimiter=',')
        b4 = np.loadtxt(b4_name, delimiter=',')

        self.b = [b1, b2, b3, b4]

        self.nlayer = len(self.W)

        self.layers = []

        for i in range(self.nlayer):
            layer = nn.Linear(self.W[i].shape[1], self.W[i].shape[0])
            layer.weight = nn.Parameter(torch.tensor(self.W[i]))
            layer.bias = nn.Parameter(torch.tensor(self.b[i]))
            self.layers.append(layer)
        
        self.bound = 1
        N = block_diag(*self.W)
        Nux = np.zeros((self.nu, self.nx))
        Nuw = N[-self.nu:, self.nx:]
        Nub = self.b[-1].reshape(self.nu, self.nu)
        Nvx = N[:-self.nu, :self.nx]
        Nvw = N[:-self.nu, self.nx:]
        Nvb = np.concatenate([self.b[0], self.b[1], self.b[2]], axis=0).reshape(self.nphi, self.nu)

        self.N = [Nux, Nuw, Nub, Nvx, Nvw, Nvb]

        R = np.linalg.inv(np.eye(*Nvw.shape) - Nvw)
        self.R = R
        Rw = Nux + Nuw @ R @ Nvx
        self.Rw = Rw
        Rb = Nuw @ R @ Nvb + Nub
        self.Rb = Rb

        xstar = np.linalg.inv(np.eye(self.A.shape[0]) - self.A - self.B @ Rw) @ self.B @ Rb
        self.xstar = xstar

        wstar = R @ Nvx @ xstar + R @ Nvb
        wstar1 = wstar[:32]
        wstar2 = wstar[32:64]
        wstar3 = wstar[64:]
        self.wstar = [wstar1, wstar2, wstar3]

        self.tanh = nn.Tanh()
    
    def step(self, input):

        u = np.clip(input, -1, 1) * self.max_torque

        theta = self.state[0]

        self.state = self.A @ self.state + self.B @ u + self.Mq * (np.sin(theta) - theta)

        self.state[2] += - self.constant_reference
        
        return self.state

    def saturation_activation(self, value):
        return torch.clamp(value, min=-self.bound, max=self.bound)
    
    def forward(self, x):

        # func = self.tanh
        func = self.saturation_activation

        nu = func(self.layers[0](torch.tensor(x)))
        nu = func(self.layers[1](nu))
        nu = func(self.layers[2](nu))
        nu = self.layers[3](nu).detach().numpy()

        return nu

    def loop(self, x0, steps):
        if self.state is None:
            self.state = x0

        states = []
        inputs = []

        for i in range(steps):
            u = self.forward(self.state)
            inputs.append(u)
            state = self.step(u)
            states.append(state)
        
        return np.array(states), inputs

if __name__ == "__main__":
    s = Integrator()
    P = np.load("P_mat.npy")

    thetalim = 20 * np.pi / 180
    vlim = 2

    inside_ROA = False
    while not inside_ROA:
        theta0 = np.random.uniform(-thetalim, thetalim)
        v0 = np.random.uniform(-vlim, vlim)
        x0 = np.array([theta0, v0, 0.0])
        if x0.T @ P @ x0 < 1:
            print(x0.T @ P @ x0)
            inside_ROA = True

    print(f"Initial state: theta0: {theta0*180/np.pi:.2f}, v0: {v0:.2f}, eta0: {0:.2f}")

    steps = 500
    states, inputs = s.loop(x0, steps)
    
    time_grid = np.linspace(0, steps, steps)

    plt.plot(time_grid, states[:, 0] - s.xstar[0])
    plt.grid(True)
    plt.show()
    
    plt.plot(time_grid, states[:, 1] - s.xstar[1])
    plt.grid(True)
    plt.show()

    plt.plot(time_grid, states[:, 2])
    plt.grid(True)
    plt.show()

    lyap = []
    xstar = s.xstar.reshape(1, 3)
    for i in range(steps):
        lyap.append((states[i].reshape(1,3) - xstar) @ P @ (states[i].reshape(1,3) - xstar).T)      
    lyap = np.squeeze(lyap)
    plt.plot(time_grid, lyap)
    plt.grid(True)
    plt.show()