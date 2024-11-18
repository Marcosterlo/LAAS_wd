from systems_and_LMI.systems.LinearPendulum import LinPendulum
import numpy as np
import os
import torch.nn as nn
import torch
from scipy.linalg import block_diag
from scipy.optimize import fsolve

# New class definition that includes non linear behavior but not the integral action
class NonLinPendulum_no_int_train(LinPendulum):
  
  def __init__(self, W, b):
    super().__init__()

    self.C = np.array([
      [0.0],
      [self.g / self.l * self.dt]
    ])

    self.nq = 1

    self.W = W
    self.b = b

    self.nlayers = 4

    self.neurons = [32, 32, 32]

    self.layers = []
    for i in range(self.nlayers):
      layer = nn.Linear(self.W[i].shape[1], self.W[i].shape[0])
      layer.weight = nn.Parameter(torch.tensor(self.W[i]))
      layer.bias = nn.Parameter(torch.tensor(self.b[i]))
      self.layers.append(layer)
    
    self.nphi = self.W[0].shape[0] + self.W[1].shape[0] + self.W[2].shape[0] + 1
    N = block_diag(*self.W)
    Nux = np.zeros((self.nu, self.nx))
    Nuw = np.concatenate([np.zeros((self.nu, self.nphi - 1)), np.array([[self.max_torque]])], axis=1)
    Nub = np.array([[0.0]])
    Nvx = N[:, :self.nx]
    Nvw = np.concatenate([N[:, self.nx:], np.zeros((self.nphi, self.nu))], axis=1)
    Nvb = np.concatenate([self.b[0], self.b[1], self.b[2], np.array([self.b[3]])], axis=0).reshape(self.nphi, self.nu)

    self.N = [Nux, Nuw, Nub, Nvx, Nvw, Nvb]

    R = np.linalg.inv(np.eye(*Nvw.shape) - Nvw)
    self.R = R
    Rw = Nux + Nuw @ R @ Nvx
    self.Rw = Rw
    Rb = Nuw @ R @ Nvb + Nub
    self.Rb = Rb

    def implicit_function(x):
      x = x.reshape(2, 1)
      I = np.eye(self.A.shape[0])
      K = np.array([[1.0, 0.0]])
      to_zero = np.squeeze((-I + self.A + self.B @ self.Rw - self.C @ K) @ x + self.C * np.sin(K @ x) + self.B @ self.Rb)
      return to_zero

    self.xstar = fsolve(implicit_function, np.array([0.0, 0.0])).reshape((self.nx, 1))

    wstar = R @ Nvx @ self.xstar + R @ Nvb
    wstar1 = wstar[:self.neurons[0]]
    wstar2 = wstar[self.neurons[0]:self.neurons[0] + self.neurons[1]]
    wstar3 = wstar[self.neurons[0] + self.neurons[1]:]
    wstar4 = wstar[-1]
    self.wstar = [wstar1, wstar2, wstar3, wstar4]  
  
  def forward(self):
    func = nn.Hardtanh()
    
    nu = func(self.layers[0](torch.tensor(self.state.reshape(1, self.nx))))
    nu = func(self.layers[1](nu))
    nu = func(self.layers[2](nu))
    nu = self.layers[3](nu).detach().numpy()

    return nu
  
  def step(self):
    u = np.clip(self.forward(), -1.0, 1.0)*self.max_torque
    nonlin = np.sin(self.state[0]) - self.state[0]
    self.state = self.A @ self.state + self.B @ u + self.C * nonlin
    return self.state, u

if __name__ == "__main__":

  W1_name = os.path.abspath(__file__ + "/../simple_weights/l1.weight.csv")
  W2_name = os.path.abspath(__file__ + "/../simple_weights/l2.weight.csv")
  W3_name = os.path.abspath(__file__ + "/../simple_weights/l3.weight.csv")
  W4_name = os.path.abspath(__file__ + "/../simple_weights/l4.weight.csv")

  W1 = np.loadtxt(W1_name, delimiter=',')
  W2 = np.loadtxt(W2_name, delimiter=',')
  W3 = np.loadtxt(W3_name, delimiter=',')
  W4 = np.loadtxt(W4_name, delimiter=',')
  W4 = W4.reshape((1, len(W4)))

  W = [W1, W2, W3, W4]

  b1_name = os.path.abspath(__file__ + "/../simple_weights/l1.bias.csv")
  b2_name = os.path.abspath(__file__ + "/../simple_weights/l2.bias.csv")
  b3_name = os.path.abspath(__file__ + "/../simple_weights/l3.bias.csv")
  b4_name = os.path.abspath(__file__ + "/../simple_weights/l4.bias.csv")
  
  b1 = np.loadtxt(b1_name, delimiter=',')
  b2 = np.loadtxt(b2_name, delimiter=',')
  b3 = np.loadtxt(b3_name, delimiter=',')
  b4 = np.loadtxt(b4_name, delimiter=',')
  
  b = [b1, b2, b3, b4] 

  s = NonLinPendulum_no_int_train(W, b)