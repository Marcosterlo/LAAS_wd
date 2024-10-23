from systems_and_LMI.systems.NonLinearPendulum import NonLinPendulum
import os
import numpy as np
import torch.nn as nn
import torch
from scipy.linalg import block_diag

# New class definition that includes the NN controller
class NonLinPendulum_NN(NonLinPendulum):
  
  def __init__(self):
    super().__init__()

    # VERY IMPORTANT PARAMETER, the training has been done with nn.Hardtanh that works with -1 and 1. If you want to change it you need to change the whole controller and train it again
    self.bound = 1
    
    # Name of csv files for weights and biases with relative import
    W1_name = os.path.abspath(__file__ + '/../nonlinear_weights/l1.weight.csv')
    W2_name = os.path.abspath(__file__ + '/../nonlinear_weights/l2.weight.csv')
    W3_name = os.path.abspath(__file__ + '/../nonlinear_weights/l3.weight.csv')
    W4_name = os.path.abspath(__file__ + '/../nonlinear_weights/l4.weight.csv')

    W1 = np.loadtxt(W1_name, delimiter=',')
    W2 = np.loadtxt(W2_name, delimiter=',')
    W3 = np.loadtxt(W3_name, delimiter=',')
    W4 = np.loadtxt(W4_name, delimiter=',')
    W4 = W4.reshape(self.nu, len(W4))

    self.W = [W1, W2, W3, W4]

    b1_name = os.path.abspath(__file__ + '/../nonlinear_weights/l1.bias.csv')
    b2_name = os.path.abspath(__file__ + '/../nonlinear_weights/l2.bias.csv')
    b3_name = os.path.abspath(__file__ + '/../nonlinear_weights/l3.bias.csv')
    b4_name = os.path.abspath(__file__ + '/../nonlinear_weights/l4.bias.csv')

    b1 = np.loadtxt(b1_name, delimiter=',')
    b2 = np.loadtxt(b2_name, delimiter=',') 
    b3 = np.loadtxt(b3_name, delimiter=',')
    b4 = np.loadtxt(b4_name, delimiter=',')

    self.b = [b1, b2, b3, b4]

    self.layers = []

    # Number of layers + 1
    self.nlayers = 4
    
    # Number of neurons per layer
    self.neurons = [32, 32, 32]

    # Creation of NN layers
    for i in range(self.nlayer):
      layer = nn.Linear(self.W[i].shape[1], self.W[i].shape[0])
      layer.weight = nn.Parameter(torch.tensor(self.W[i]))
      layer.bias = nn.Parameter(torch.tensor(self.b[i]))
      self.layers.append(layer)

    # Creation of the N matrix
    self.nphi = W1.shape[0] + W2.shape[0] + W3.shape[0]
    N = block_diag(*self.W)
    Nux = np.zeros((self.nu, self.nx))
    Nuw = N[-self.nu:, self.nx:]
    Nub = self.b[-1].reshape(self.nu, self.nu)
    Nvx = N[:-self.nu, :self.nx]
    Nvw = N[:-self.nu, self.nx:]
    Nvb = np.concatenate([self.b[0], self.b[1], self.b[2]], axis=0).reshape(self.nphi, self.nu)

    self.N = [Nux, Nuw, Nub, Nvx, Nvw, Nvb]

    # Creation of auxiliary matrices
    R = np.linalg.inv(np.eye(*Nvw.shape) - Nvw)
    self.R = R
    Rw = Nux + Nuw @ R @ Nvx
    self.Rw = Rw
    Rb = Nuw @ R @ Nvb + Nub
    self.Rb = Rb

    # Equilibria computation    
    xstar = np.linalg.inv(np.eye(self.A.shape[0]) - self.A - self.B @ Rw) @ self.B @ Rb
    self.xstar = xstar

    wstar = R @ Nvx @ xstar + R @ Nvb
    wstar1 = wstar[:32]
    wstar2 = wstar[32:64]
    wstar3 = wstar[64:]
    self.wstar = [wstar1, wstar2, wstar3]

  
  def forward(self):
    
    # Activation function definition, saturation in this case
    func = nn.Hardtanh()

    nu = func(self.layers[0](torch.tensor(self.state.reshape(1, 3))))
    nu = func(self.layers[1](nu))
    nu = func(self.layers[2](nu))
    # Activation function not applied to the last layer
    nu = self.layers[3](nu).detach().numpy()

    return nu
  
  def step(self):
    # Compute the control input
    u = self.forward()
    # Compute the non-linear term
    nonlin = np.sin(self.state[0]) - self.state[0]
    # Compute the state update
    self.state = self.A @ self.state + self.B @ u + self.C * nonlin
    # Adds the constant reference to the integral term
    self.state[2] += -self.constant_reference
    return self.state, u
  
if __name__ == "__main__":

  s = NonLinPendulum_NN()