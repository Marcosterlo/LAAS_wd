import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.linalg import block_diag

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

    # Neurons per layer in the FFNN (to substitute with array if different)
    neurons = 32*np.ones((1,2)).astype(np.int16)
    self.neurons = neurons.tolist()[0]

    # Gain matrix
    self.K = np.array([-0.1, 0]).reshape(1,2)

    # Equilibrium states
    N = block_diag(*self.W) # Reminder for myself: * operator unpacks lists and pass it as singular arguments

    Nux = self.K
    Nuw = N[self.nphi:, self.nx:]
    Nub = self.b[-1]

    Nvx = N[:self.nphi, :self.nx]
    Nvw = N[:self.nphi, self.nx:]
    Nvb = np.concatenate((self.b[0], self.b[1]))
    R = np.linalg.inv(np.eye(*Nvw.shape) - Nvw)
    Rw = Nux + Nuw @ R @ Nvx
    Rb = Nuw @ R @ Nvb + Nub
    self.xstar = np.linalg.inv(np.eye(self.A.shape[0]) - self.A - self.B @ Rw) @ self.B @ Rb
    wstar = R @ Nvx @ self.xstar + R @ Nvb
    wstar = np.split(wstar, [32])[0]

    # Std of random state noise
    self.stdx = 0.01

    # Std of random input noise
    self.stdu = 0.01

   
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

    self.layers = [self.layer1, self.layer2, self.layer3]

    # T and Z matrices import from LMI solution
    T = np.load("T_mat.npy")
    Z = np.load("Z_mat.npy")
    G = np.linalg.inv(T) @ Z
    indeces = np.linspace(0, self.nphi, self.nlayer).astype(np.int8)
    indeces = indeces[1:-1].tolist()
    self.G = np.split(G, indeces)
    self.T = []
    for i in range(self.nlayer -1):
      self.T.append(T[i*self.neurons[i]:(i+1)*self.neurons[i], i*self.neurons[i]:(i+1)*self.neurons[i]])

    ## Class variable
    self.state = None
    self.last_w = []
    self.wstar = []
    for i, neuron in enumerate(self.neurons):
      self.last_w.append(np.ones((neuron, 1))*1e3)
      self.wstar.append(np.ones((neuron, 1))*wstar[i])
    
  # Function that predicts the input
  def forward(self, x):
    e1 = 0
    e2 = 0
    # 1st layer
    nu = self.layers[0](x).detach().numpy().reshape(32, 1)

    # ETM evaluation
    vec1 = (nu - self.last_w[0]).T
    T = self.T[0]
    vec2 = (self.G[0] @ (x.detach().numpy() - self.xstar).reshape(2,1) - (self.last_w[0] - self.wstar[0]))

    if (vec1 @ T @ vec2 > 0):
    # if True:
      # If there is an event we compute the output of the layer with the non linear activation function
      omega = self.saturation_activation(torch.tensor(nu))
      # We substitute last computed value
      self.last_w[0] = omega.detach().numpy()
      e1 = 1
    else:
      # If no event occurs we feed the next layer the last stored output
      omega = self.last_w[0]

    # 2nd layer
    nu = self.layers[1](torch.tensor(omega.T)).detach().numpy().reshape(32, 1)

    # 2nd ETM evaluation
    vec1 = (nu - self.last_w[1]).T
    T = self.T[1]
    vec2 = (self.G[1] @ (x.detach().numpy() - self.xstar).reshape(2,1) - (self.last_w[1] - self.wstar[1]))

    if (vec1 @ T @ vec2 > 0):
    # if True:
      omega = self.saturation_activation(torch.tensor(nu))
      self.last_w[1] = omega.detach().numpy()
      e2 = 1
    else:
      # If no event occurs we feed the next layer the last stored output
      omega = self.last_w[1]

    # 3rd layer
    omega = self.layers[2](torch.tensor(omega.T))

    return omega, e1, e2

  # Customly defined activation function since sat doesn't exist on tensorflow
  def saturation_activation(self, value):
    return torch.clamp(value, min=-self.bound, max=self.bound)

  def dynamic_loop(self, x0, nstep):

    e = np.zeros((2, nstep))

    if self.state == None:
      self.state = x0

    states = []
    inputs = []

    for i in range(nstep):
      u, e1, e2 = self.forward(torch.tensor(self.state.reshape(1, 2)))
      u = u.detach().numpy()[0]
      x = (self.A + self.B @ self.K) @ self.state.reshape(2,1) + self.B @ u.reshape(1, 1)

      self.state = x.reshape(2, 1)# + (np.random.randn(self.nx)*self.stdx).reshape(2, 1)
      u = u# + np.random.randn(self.nu)*self.stdu

      states.append(x)
      inputs.append(u)

      if e1:
        e[0, i] = 1

      if e2:
        e[1, i] = 1 

    return np.array(states), np.array(inputs), e
  

# Main execution
if __name__ == "__main__":

  # Test of instance creation
  s = System()

  P = np.load("P_mat.npy")

  states = []
  inputs = []
  events = []
  lyap = []

  x0 = np.array([1.1, -2.9])
  # x0 = s.xstar
  nstep = 150 

  states, inputs, events = s.dynamic_loop(x0, nstep)
  x = states[:, 0]
  v = states[:, 1]
  u = inputs.reshape(len(inputs))
  time_grid = np.linspace(0, nstep * s.dt, nstep)
  
  for i in range(nstep):
    lyap.append(((states[i, :] - s.xstar).T @ P @ (states[i, :] - s.xstar))[0][0])

  for i, event in enumerate(events[1, :]):
    if not event:
      events[1, i] = None
  for i, event in enumerate(events[0, :]):
    if not event:
      events[0, i] = None

  fig, axs = plt.subplots(2, 2)
  axs[0, 0].plot(time_grid, x - s.xstar[0])
  axs[0, 0].plot(time_grid, events[1, :]*(x - s.xstar[0]).reshape(len(time_grid)), 'ro')
  axs[0, 0].set_title("Position")
  axs[0, 0].set_xlabel("Time [s]")
  axs[0, 0].set_ylabel("Position [rad]")

  axs[0, 1].plot(time_grid, v - s.xstar[1])
  axs[0, 1].plot(time_grid, events[1, :]*(v - s.xstar[1]).reshape(len(time_grid)), 'ro')
  axs[0, 1].set_title("Velocity")
  axs[0, 1].set_xlabel("Time [s]")
  axs[0, 1].set_ylabel("Velocity [rad/s]")

  axs[1, 0].plot(time_grid, u)
  axs[1, 0].plot(time_grid, events[1, :]*u, 'ro')
  axs[1, 0].set_title("Inputs")
  axs[1, 0].set_xlabel("Time [s]")
  axs[1, 0].set_ylabel("Control input")

  axs[1, 1].plot(time_grid, events[1, :]*lyap, 'ro')
  axs[1, 1].plot(time_grid, lyap)
  axs[1, 1].set_title("Lyapunov function")

  plt.show()
