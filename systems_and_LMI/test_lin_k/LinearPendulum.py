import numpy as np
import os
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

class LinearPendulum():

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

    b1_name = os.path.abspath(__file__ + "/../weights/b1.csv")
    b2_name = os.path.abspath(__file__ + "/../weights/b2.csv")
    b3_name = os.path.abspath(__file__ + "/../weights/b3.csv")
    b4_name = os.path.abspath(__file__ + "/../weights/b4.csv")

    b1 = np.loadtxt(b1_name, delimiter=',')
    b2 = np.loadtxt(b2_name, delimiter=',')
    b3 = np.loadtxt(b3_name, delimiter=',')
    b4 = np.loadtxt(b4_name, delimiter=',')

    self.b = [b1, b2, b3, b4]

    self.layers = []

    self.nlayer = 4
    for i in range(self.nlayer):
      layer = nn.Linear(self.W[i].shape[1], self.W[i].shape[0])
      layer.weight = nn.Parameter(torch.tensor(self.W[i]))
      layer.bias = nn.Parameter(torch.tensor(self.b[i]))
      self.layers.append(layer)
    
    self.nphi = W1.shape[0] + W2.shape[0] + W3.shape[0]
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

    self.neurons = [32, 32, 32]

  
  def forward(self):
    func = nn.Hardtanh()

    nu = func(self.layers[0](torch.tensor(self.state.reshape(1, 3))))
    nu = func(self.layers[1](nu))
    nu = func(self.layers[2](nu))
    nu = self.layers[3](nu).detach().numpy()

    return nu
  
  def step(self):

    u = self.forward()
    newx = self.A @ self.state + self.B @ u
    newx[2] += -self.constant_reference
    self.state = newx
    return self.state, u

if __name__ == "__main__":

  s = LinearPendulum()

  P = np.load("P.npy")

  ellip = []

  for i in range(1000000):
    x0 = np.array([[np.random.uniform(-20, 20)], [np.random.uniform(-20, 20)], [0.0]])
    if ((x0 - s.xstar).T @ P @ (x0 - s.xstar) <= 1):
      ellip.append(x0)
  
  ellip = np.squeeze(np.array(ellip))

  theta_lim = 90*np.pi/180
  vtheta_lim = 5

  n_steps = 300
  n_trials = 20

  x0 = np.array([[np.pi/2], [1.0], [0.0]])
  
  for i in range(n_trials):
    theta0 = np.random.uniform(-theta_lim, theta_lim)
    vtheta0 = np.random.uniform(-vtheta_lim, vtheta_lim)
    x0 = np.array([[theta0], [vtheta0], [0.0]])
    s.state = x0

    print(f"Initial state: theta0: {theta0*180/np.pi:.2f}, v0: {vtheta0:.2f}, eta0: {0:.2f}")
    print(f"Is initial state inside ellipsoid? {((x0 - s.xstar).T @ P @ (x0 - s.xstar) <= 1)[0][0]}")

    states = []
    inputs = []
    lyap = []

    for k in range(n_steps):
      state, u = s.step()
      states.append(state)
      inputs.append(u)
      lyap.append((state - s.xstar).T @ P @ (state - s.xstar))
    
    states = np.array(states)
    inputs = np.array(inputs)
    timegrid = np.linspace(0, n_steps, n_steps)

    # plt.plot(timegrid, states[:, 0])
    # plt.title("Theta")
    # plt.grid(True)
    # plt.show()

    # plt.plot(timegrid, states[:, 1])
    # plt.title("Vtheta")
    # plt.grid(True)
    # plt.show()

    # plt.plot(timegrid, states[:, 2])
    # plt.title("Eta")
    # plt.grid(True)
    # plt.show()

    plt.plot(ellip[:, 0], ellip[:, 1], 'ro')
    plt.plot(x0[0], x0[1], 'bo', markersize=10)
    plt.plot(states[:, 0], states[:, 1])
    plt.grid(True)
    plt.show()

    plt.plot(timegrid, np.squeeze(lyap))
    plt.title("Lyapunov function")
    plt.grid(True)
    plt.show()
