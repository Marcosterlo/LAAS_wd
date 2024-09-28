import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path

class System:
  
  def __init__(self):
    
    ## Parameters
    self.g = 9.81 # Gravity coefficient
    self.m = 0.15 # mass
    self.l = 0.5 # length
    self.mu = 0.05 # frict coeff
    self.dt = 0.02 # sampling period

    ## System definition x^+ = A*x + B*u

    self.A = np.array([[1,      self.dt],
                  [self.g/self.l*self.dt, 1 - self.mu/(self.m*self.l**2)*self.dt]])

    self.B = np.array([[0],
                  [self.dt/(self.m*self.l**2)]])

    # Size of vector x
    self.nx = 2

    # Size of input u
    self.nu = 1

    # Std of random state noise
    self.stdx = 0.01

    # Std of random input noise
    self.stdu = 0.01

    # Weights import of trained FFNN
    mat_name = Path(__file__).with_name('weight_saturation.mat')
    data = scipy.io.loadmat(mat_name)
    W1 = data['W1']
    W2 = data['W2']
    W3 = data['W3']

    # Vector of weigts
    self.W = [W1, W2, W3]

    # Number of neurons
    self.nphi = W1.shape[0] + W2.shape[0]
    
    # Bias import of trained FFNN
    b1 = data['b1']
    b2 = data['b2']
    b3 = data['b3']
    b1 = b1.reshape(len(b1))
    b2 = b2.reshape(len(b2))
    b3 = b3.reshape(len(b3))

    # Vector of biases
    self.b = [b1, b2, b3]
   
    # Number of layers of the FFNN
    self.nlayer = 3

    # Bound value for saturation function
    self.bound = 1

    # Model cereation
    self.layer1 = nn.Linear(self.nx, W1.shape[0])
    self.layer2 = nn.Linear(W1.shape[0], W2.shape[0])
    self.layer3 = nn.Linear(W2.shape[0], self.nu)

    # Weights import
    self.layer1.weight = nn.Parameter(torch.tensor(W1))
    self.layer2.weight = nn.Parameter(torch.tensor(W2))
    self.layer3.weight = nn.Parameter(torch.tensor(W3))

    self.layer1.bias = nn.Parameter(torch.tensor(b1))
    self.layer2.bias = nn.Parameter(torch.tensor(b2))
    self.layer3.bias = nn.Parameter(torch.tensor(b3))


    ## Class variable
    self.state = 0
  
  # Function that predicts the input
  def forward(self, x):
    x = self.saturation_activation(self.layer1(x))
    x = self.saturation_activation(self.layer2(x))
    x = self.saturation_activation(self.layer3(x))
    return x

  # Customly defined activation function since sat doesn't exist on tensorflow
  def saturation_activation(self, value):
    return torch.clamp(value, min=-self.bound, max=self.bound)

  def dynamic_loop(self, x0, nstep):

    if self.state == 0:
      self.state = x0

    states = []
    inputs = []

    for i in range(nstep):
      u = self.forward(torch.tensor(self.state)).detach().numpy()
      x = self.A @ self.state.T + self.B @ u

      self.state = x.T + np.random.randn(self.nx)*self.stdx
      u = u + np.random.randn(self.nu)*self.stdu

      states.append(x)
      inputs.append(u)
    
    return np.array(states), np.array(inputs)
  

# Main execution
if __name__ == "__main__":

  # Test of instance creation
  s = System()

  states = []
  inputs = []

  x0 = np.array([[1.1, -2.9]])
  nstep = 300

  states, inputs = s.dynamic_loop(x0, nstep)
  x = states[:, 0]
  v = states[:, 1]
  u = inputs.reshape(len(inputs))
  time_grid = np.linspace(0, nstep * s.dt, nstep)

  fig, axs = plt.subplots(3, 1)
  axs[0].plot(time_grid, x)
  axs[0].set_title("Position")
  axs[0].set_xlabel("Time [s]")
  axs[0].set_ylabel("Position [rad]")

  axs[1].plot(time_grid, v)
  axs[1].set_title("Velocity")
  axs[1].set_xlabel("Time [s]")
  axs[1].set_ylabel("Velocity [rad/s]")

  axs[2].plot(time_grid, u)
  axs[2].set_title("Inputs")
  axs[2].set_xlabel("Time [s]")
  axs[2].set_ylabel("Control input")

  plt.show()
