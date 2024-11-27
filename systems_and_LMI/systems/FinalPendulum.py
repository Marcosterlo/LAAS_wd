import numpy as np
import torch.nn as nn
import torch
import warnings
from scipy.linalg import block_diag
from scipy.optimize import fsolve

warnings.filterwarnings("ignore", category=UserWarning, module='torch')

class FinalPendulum():
  
  def __init__(self, W, b, ref):
    
    warnings.filterwarnings("ignore", category=UserWarning)

    self.state = None
    self.g = 9.81
    self.m = 0.15
    self.l = 0.5
    self.mu = 0.05
    self.dt = 0.02
    self.max_torque = 5
    self.max_speed = 8.0
    self.constant_reference = ref
    self.nx = 3
    self.nu = 1
    self.nq = 1
    self.bound = 1
    self.nlayers = 4
    self.layers = []
    self.W = W
    self.b = b
    self.nphi = self.W[0].shape[0] + self.W[1].shape[0] + self.W[2].shape[0]
    self.neurons = [32, 32, 32]

    self.A = np.array([
        [1, self.dt, 0],
        [self.g*self.dt/self.l, 1-self.mu*self.dt/(self.m*self.l**2), 0],
        [1, 0, 1]
    ])

    self.B = np.array([
        [0],
        [self.dt/(self.m*self.l**2) * self.max_torque],
        [0]
    ])

    self.C = np.array([
      [0],
      [self.g / self.l * self.dt],
      [0]
    ])

    self.D = np.array([
      [0],
      [0],
      [-1]
    ])

    for i in range(self.nlayers):
      layer = nn.Linear(self.W[i].shape[1], self.W[i].shape[0])
      layer.weight = nn.Parameter(torch.tensor(self.W[i]))
      layer.bias = nn.Parameter(torch.tensor(self.b[i]))
      self.layers.append(layer)

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
    
    def implicit_function(x):
      x = x.reshape(self.nx, 1)
      I = np.eye(self.A.shape[0])
      K = np.array([[1.0, 0.0, 0.0]])
      to_zero = np.squeeze((-I + self.A + self.B @ self.Rw - self.C @ K) @ x + self.C * np.sin(K @ x) + self.D * self.constant_reference + self.B @ self.Rb)
      return to_zero
    
    self.xstar = fsolve(implicit_function, np.array([[self.constant_reference], [0.0], [0.0]])).reshape(self.nx, 1)

    wstar = R @ Nvx @ self.xstar + R @ Nvb
    wstar1 = wstar[:self.neurons[0]]
    wstar2 = wstar[self.neurons[0]:self.neurons[0] + self.neurons[1]]
    wstar3 = wstar[self.neurons[0] + self.neurons[1]:self.neurons[0] + self.neurons[1] + self.neurons[2]]
    self.wstar = [wstar1, wstar2, wstar3]

  def forward(self):
    func = nn.Hardtanh()
    nu = func(self.layers[0](torch.tensor(self.state.reshape(1, self.nx))))
    nu = func(self.layers[1](nu))
    nu = func(self.layers[2](nu))
    nu = self.layers[3](nu).detach().numpy()

    return nu
  
  def step(self):
    u = np.clip(self.forward(), -1.0, 1.0)
    nonlin = np.sin(self.state[0]) - self.state[0]
    self.state = self.A @ self.state + self.B @ u + self.C * nonlin + self.D * self.constant_reference
    return self.state, u

if __name__ == "__main__":
  import os

  W1_name = os.path.abspath(__file__ + "/../nonlin_norm_weights/mlp_extractor.policy_net.0.weight.csv")
  W2_name = os.path.abspath(__file__ + "/../nonlin_norm_weights/mlp_extractor.policy_net.2.weight.csv")
  W3_name = os.path.abspath(__file__ + "/../nonlin_norm_weights/mlp_extractor.policy_net.4.weight.csv")
  W4_name = os.path.abspath(__file__ + "/../nonlin_norm_weights/action_net.weight.csv")
  
  b1_name = os.path.abspath(__file__ + "/../nonlin_norm_weights/mlp_extractor.policy_net.0.bias.csv")
  b2_name = os.path.abspath(__file__ + "/../nonlin_norm_weights/mlp_extractor.policy_net.2.bias.csv")
  b3_name = os.path.abspath(__file__ + "/../nonlin_norm_weights/mlp_extractor.policy_net.4.bias.csv")
  b4_name = os.path.abspath(__file__ + "/../nonlin_norm_weights/action_net.bias.csv")
  
  W1 = np.loadtxt(W1_name, delimiter=',')
  W2 = np.loadtxt(W2_name, delimiter=',')
  W3 = np.loadtxt(W3_name, delimiter=',')
  W4 = np.loadtxt(W4_name, delimiter=',')
  W4 = W4.reshape((1, len(W4)))
  
  W = [W1, W2, W3, W4]
  
  b1 = np.loadtxt(b1_name, delimiter=',')
  b2 = np.loadtxt(b2_name, delimiter=',')
  b3 = np.loadtxt(b3_name, delimiter=',')
  b4 = np.loadtxt(b4_name, delimiter=',')
  
  b = [b1, b2, b3, b4]
  
  s = FinalPendulum(W, b, -0.3)

  x0 = np.array([[np.pi/2], [3.0], [0.0]])

  s.state = x0

  n_steps = 1000

  states = []
  inputs = []

  for i in range(n_steps):
    state, u = s.step()
    states.append(state-s.xstar)
    inputs.append(u)
  
  states = np.squeeze(np.array(states))
  inputs = np.squeeze(np.array(inputs))
  
  import matplotlib.pyplot as plt
  
  plt.plot(states[:,0], states[:,1])
  plt.grid(True)
  plt.show()

  plt.plot(states[:, 2])
  plt.grid(True)
  plt.show()
  
  plt.plot(inputs)
  plt.grid(True)
  plt.show()

    