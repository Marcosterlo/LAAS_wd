import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import os
import warnings

warnings.filterwarnings('ignore')

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

    # Gain matrix
    self.K = np.array([[-0.1, 0]])

    # Size of vector x
    self.nx = 2

    # Size of input u
    self.nu = 1

    # Weights import of trained FFNN
    # Script to use relative path
  
    W1_name = os.path.abspath(__file__ + "/../4_layers/W1.csv")
    W1 = np.loadtxt(W1_name, delimiter=',')
    W2_name = os.path.abspath(__file__ + "/../4_layers/W2.csv")
    W2 = np.loadtxt(W2_name, delimiter=',')
    W3_name = os.path.abspath(__file__ + "/../4_layers/W3.csv")
    W3 = np.loadtxt(W3_name, delimiter=',')
    W4_name = os.path.abspath(__file__ + "/../4_layers/W4.csv")
    W4 = np.loadtxt(W4_name, delimiter=',')
    W4 = W4.reshape(self.nu, len(W4))

    # Vector of weigts
    self.W = [W1, W2, W3, W4]

    # Number of neurons
    self.nphi = W1.shape[0] + W2.shape[0] + W3.shape[0]
    
    # Bias import of trained FFNN
    b1_name = os.path.abspath(__file__ + "/../4_layers/b1.csv")
    b1 = np.loadtxt(b1_name, delimiter=',')
    b2_name = os.path.abspath(__file__ + "/../4_layers/b2.csv")
    b2 = np.loadtxt(b2_name, delimiter=',')
    b3_name = os.path.abspath(__file__ + "/../4_layers/b3.csv")
    b3 = np.loadtxt(b3_name, delimiter=',')
    b4_name = os.path.abspath(__file__ + "/../4_layers/b4.csv")
    b4 = np.loadtxt(b4_name, delimiter=',')

    # Vector of biases
    self.b = [b1, b2, b3, b4]

    self.nlayer = len(self.W)

    # Neurons per layer in the FFNN (to substitute with array if different)
    neurons = 32*np.ones((1, self.nlayer - 1)).astype(np.int16)
    self.neurons = neurons.tolist()[0]

    # Model creation

    self.layers = []

    for i in range(self.nlayer):
      layer = nn.Linear(self.W[i].shape[1], self.W[i].shape[0])
      layer.weight = nn.Parameter(torch.tensor(self.W[i]))
      layer.bias = nn.Parameter(torch.tensor(self.b[i]))
      self.layers.append(layer)

    # Bound value for saturation function
    self.bound = 1

    # Closed loop matrices
    N = block_diag(*self.W) # Reminder for myself: * operator unpacks lists and pass it as singular arguments
    Nux = self.K
    Nuw = N[-self.nu:, self.nx:]
    Nub = self.b[-1].reshape(self.nu, self.nu)
    Nvx = N[:-self.nu, :self.nx]
    Nvw = N[:-self.nu, self.nx:]
    Nvb = np.array([[self.b[0]], [self.b[1]], [self.b[2]]]).reshape(self.nphi, self.nu)

    self.N = [Nux, Nuw, Nub, Nvx, Nvw, Nvb]

    # Auxiliary matrices
    R = np.linalg.inv(np.eye(*Nvw.shape) - Nvw)
    Rw = Nux + Nuw @ R @ Nvx
    Rb = Nuw @ R @ Nvb + Nub

    # Equilibrium states
    self.xstar = np.linalg.inv(np.eye(self.A.shape[0]) - self.A - self.B @ Rw) @ self.B @ Rb
    wstar = R @ Nvx @ self.xstar + R @ Nvb

    # I create the array indeces to split wstar into
    self.wstar = np.split(wstar, self.nlayer - 1)

    ## Class variable
    self.state = None

    ## Function that predicts the input
  def forward(self, x):

    # Reshape fo state according to NN dimensions
    x = x.reshape(1, self.W[0].shape[1])

    nu = self.saturation_activation(self.layers[0](torch.tensor(x)))
    nu = self.saturation_activation(self.layers[1](nu))
    nu = self.saturation_activation(self.layers[2](nu))
    nu = self.layers[3](nu).detach().numpy()

    return nu

  ## Customly defined activation function since sat doesn't exist on tensorflow
  def saturation_activation(self, value):
    return torch.clamp(value, min=-self.bound, max=self.bound)

  ## Function that evaluates the closed loop dynamics
  def dynamic_loop(self, x0, nstep):

    # Initializes the state variable of the system to the initial condition
    if self.state is None:
      self.state = x0

    states = []
    inputs = []

    # Loop called at each step
    for i in range(nstep):

      # Input computation via NN controller, events are tracked
      u = self.forward(self.state)
      
      # Forward dynamics
      x = (self.A + self.B @ self.K) @ self.state.reshape(2,1) + self.B @ u.reshape(1, 1)

      # Optional noise addition
      # self.state = x + (np.random.randn(self.nx)*self.stdx).reshape(2, 1)
      self.state = x
      # u = u + np.random.randn(self.nu)*self.stdu

      states.append(x)
      inputs.append(u)

    return np.array(states), np.array(inputs)
  

# Main execution
if __name__ == "__main__":

  # Systam object creation
  s = System()

  check_lyap = False

  if check_lyap:
    P = np.load('./4_layers/P_mat.npy')

  n_test = 20

  for i in range(n_test):
    
    # Empty vectors initialization
    states = []
    inputs = []
    if check_lyap:
      lyap = []

    # Simulation parameters
    if check_lyap:
      not_in_ellipsoid = True
      while not_in_ellipsoid:
        x1 = np.random.uniform(-np.pi/2, np.pi/2)
        x2 = np.random.uniform(-5, 5)
        x0 = np.array([x1, x2])
        if (x0.T @ P @ x0 < 1): 
          not_in_ellipsoid = False
    else:
        x1 = np.random.uniform(-np.pi/4, np.pi/4)
        x2 = np.random.uniform(-1, 1)
        x0 = np.array([x1, x2])

    x1_deg = x1/np.pi*180
    print(f"Initial state: position: {x1_deg:.2f}°, speed: {x2:.2f} [rad/s]")

    nstep = 300
    # Call to simulation function of object System
    s.state = None
    states, inputs = s.dynamic_loop(x0, nstep)

    # Unpacking vectors
    x = states[:, 0]
    v = states[:, 1]
    u = inputs.reshape(len(inputs))
    time_grid = np.linspace(0, nstep * s.dt, nstep)

    if check_lyap:
      for i in range(nstep):
        lyap.append((states[i] - s.xstar).T @ P @ (states[i] - s.xstar))
      lyap = np.array(lyap).reshape(nstep)
    
    ## Plotting
    
    if check_lyap:
      fig, axs = plt.subplots(4)
    else:
      fig, axs = plt.subplots(3)

    axs[0].plot(time_grid, x - s.xstar[0])
    axs[0].set_ylabel('x1')
    axs[0].grid(True)

    axs[1].plot(time_grid, v - s.xstar[1])
    axs[1].set_ylabel('x2')
    axs[1].grid(True)

    axs[2].plot(time_grid, u)
    axs[2].set_ylabel('u')
    axs[2].grid(True)

    if check_lyap:
      axs[3].plot(time_grid, lyap)
      axs[3].set_ylabel('lyap')
      axs[3].grid(True)

    plt.show()

