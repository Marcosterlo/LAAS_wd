from systems_and_LMI.systems.NonLinearPendulum import NonLinPendulum
import numpy as np
import os
import torch.nn as nn
import torch
from scipy.linalg import block_diag
from scipy.optimize import fsolve
import warnings

# User warnings filter
warnings.filterwarnings("ignore", category=UserWarning, module='torch')

# New class definition that depends on the passed weights and biases
class NonLinPendulum_train2l(NonLinPendulum):
  
  def __init__(self, W, b, ref):
    super().__init__(ref)

    warnings.filterwarnings("ignore", category=UserWarning)

    self.nq = 1
    self.nr = 1
    
    self.W = W
    self.b = b
    
    self.nlayers = 3

    self.neurons = [16, 16]

    self.layers = []
    for i in range(self.nlayers):
      layer = nn.Linear(self.W[i].shape[1], self.W[i].shape[0])
      try:
        layer.weight = nn.Parameter(self.W[i].clone().detach().requires_grad_(True))
        layer.bias = nn.Parameter(self.b[i].clone().detach().requires_grad_(True))
      except:
        layer.weight = nn.Parameter(torch.tensor(self.W[i]))
        layer.bias = nn.Parameter(torch.tensor(self.b[i]))
      self.layers.append(layer)
      
    self.nphi = 0
    for i in range(self.nlayers - 1):
      self.nphi += self.W[i].shape[0]
    self.nphi += 1

    N = block_diag(*self.W)
    Nux = np.zeros((self.nu, self.nx))
    Nuw = np.concatenate([np.zeros((self.nu, self.nphi - 1)), np.array([[self.max_torque]])], axis=1)
    Nub = np.array([[0.0]])
    Nvx = N[:, :self.nx]
    Nvw = np.concatenate([N[:, self.nx:], np.zeros((self.nphi, self.nu))], axis=1)
    try:
      Nvb = np.concatenate([self.b[0], self.b[1], np.array([self.b[2]])], axis=0).reshape(self.nphi, self.nu)
    except:
      Nvb = np.concatenate([self.b[0], self.b[1], self.b[2]], axis=0).reshape(self.nphi, self.nu)
      
    self.N = [Nux, Nuw, Nub, Nvx, Nvw, Nvb]
    
    R = np.linalg.inv(np.eye(*Nvw.shape) - Nvw)
    self.R = R
    Rw = Nux + Nuw @ R @ Nvx
    self.Rw = Rw
    Rb = Nuw @ R @ Nvb + Nub
    self.Rb = Rb
    
    def implicit_function(x):
      x = x.reshape(3, 1)
      I = np.eye(self.A.shape[0])
      K = np.array([[1.0, 0.0, 0.0]])
      to_zero = np.squeeze((-I + self.A + self.B @ self.Rw - self.C @ K) @ x + self.C * np.sin(K @ x) + self.D * self.constant_reference + self.B @ self.Rb)
      return to_zero

    self.xstar = fsolve(implicit_function, np.array([[self.constant_reference], [0.0], [0.0]])).reshape(3,1)

    wstar = R @ Nvx @ self.xstar + R @ Nvb
    self.wstar = []
    offset = 0
    for i in range(self.nlayers - 1):
      self.wstar.append(wstar[offset:offset + self.neurons[i]])
      offset += self.neurons[i]
    self.wstar.append(wstar[-1])

  def forward(self):
    func = nn.Hardtanh()
    
    nu = func(self.layers[0](torch.tensor(self.state.reshape(1, self.nx))))
    nu = func(self.layers[1](nu))
    nu = self.layers[2](nu).detach().numpy()
    return nu
    
  def step(self):
    u = np.clip(self.forward(), -1.0, 1.0)*self.max_torque
    nonlin = np.sin(self.state[0]) - self.state[0]
    self.state = self.A @ self.state + self.B @ u + self.C * nonlin + self.D * self.constant_reference
    return self.state, u
      
if __name__ == "__main__":

  W1_name = os.path.abspath(__file__ + "/../nonlin_2l/l1.weight.csv")
  W2_name = os.path.abspath(__file__ + "/../nonlin_2l/l2.weight.csv")
  W3_name = os.path.abspath(__file__ + "/../nonlin_2l/l3.weight.csv")

  b1_name = os.path.abspath(__file__ + "/../nonlin_2l/l1.bias.csv")
  b2_name = os.path.abspath(__file__ + "/../nonlin_2l/l2.bias.csv")
  b3_name = os.path.abspath(__file__ + "/../nonlin_2l/l3.bias.csv")
  
  W1 = np.loadtxt(W1_name, delimiter=',')
  W2 = np.loadtxt(W2_name, delimiter=',')
  W3 = np.loadtxt(W3_name, delimiter=',')
  W3 = W3.reshape((1, len(W3)))

  b1 = np.loadtxt(b1_name, delimiter=',')
  b2 = np.loadtxt(b2_name, delimiter=',')
  b3 = np.loadtxt(b3_name, delimiter=',')
  
  W = [W1, W2, W3]
  b = [b1, b2, b3]
  
  s = NonLinPendulum_train2l(W, b, -0.3)

  x0 = np.array([[np.pi/2], [3.0], [0.0]])

  s.state = x0
  
  n_steps = 1000

  states = []
  inputs = []
  
  for i in range(n_steps):
    state, u = s.step()
    states.append(state - s.xstar)
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