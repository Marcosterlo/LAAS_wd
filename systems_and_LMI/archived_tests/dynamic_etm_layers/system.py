import numpy as np
import scipy.io
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import os
import sys
import params

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
    # Script to use relative path
    mat_name = os.path.abspath(__file__ + "/../mat-weights/weight_saturation.mat")
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
    self.K = np.array([[-0.1, 0]])

    # Closed loop matrices
    N = block_diag(*self.W) # Reminder for myself: * operator unpacks lists and pass it as singular arguments
    Nux = self.K
    Nuw = N[self.nphi:, self.nx:]
    Nub = self.b[-1]
    Nvx = N[:self.nphi, :self.nx]
    Nvw = N[:self.nphi, self.nx:]
    Nvb = np.concatenate((self.b[0], self.b[1]))

    self.N = [Nux, Nuw, Nub, Nvx, Nvw, Nvb]

    # Auxiliary matrices
    R = np.linalg.inv(np.eye(*Nvw.shape) - Nvw)
    self.R = R
    Rw = Nux + Nuw @ R @ Nvx
    self.Rw = Rw
    Rb = Nuw @ R @ Nvb + Nub
    self.Rb = Rb

    # Equilibrium states
    self.xstar = np.linalg.inv(np.eye(self.A.shape[0]) - self.A - self.B @ Rw) @ self.B @ Rb
    wstar = R @ Nvx @ self.xstar + R @ Nvb

    # I create the array indeces to split wstar into
    indeces = [self.neurons[0]]
    self.wstar = np.split(wstar, indeces)

    # Std of random state noise
    self.stdx = 0.01

    # Std of random input noise
    self.stdu = 0.01
   
    # Number of layers of the FFNN (+1 to include last layer for input computation)
    self.nlayer = 3

    # Bound value for saturation function
    self.bound = 1

    # Model creation
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

    # T and Z matrices import from LMI solution with relative path
    T_mat_name = os.path.abspath(__file__ + "/../mat-weights/T_try.npy")
    Z_mat_name = os.path.abspath(__file__ + "/../mat-weights/Z_try.npy")
    T = np.load(T_mat_name)
    Z = np.load(Z_mat_name)
    # I compose G matrix following the paper
    G = np.linalg.inv(T) @ Z
    # I split it into blocks per each layer
    self.G = np.split(G, indeces)
    self.T = []
    # I split block diagonal matrix T into blocks for each layer
    for i in range(self.nlayer -1):
      self.T.append(T[i*self.neurons[i]:(i+1)*self.neurons[i], i*self.neurons[i]:(i+1)*self.neurons[i]])

    ## Class variable
    self.state = None
    self.last_w = []
    for i, neuron in enumerate(self.neurons):
      # Initialized with big values to trigger an update at first time step
      self.last_w.append(np.ones((neuron, 1))*1e3)
      # Reshape of wstar vector to make it compatible with last_w
      self.wstar[i] = self.wstar[i].reshape(len(self.wstar[i]), 1)

    self.eta = np.ones(self.nlayer - 1)*0
    self.rho = params.rhos
    self.lam = params.lambdas

    ## Function that predicts the input
  def forward(self, x, ETM, DYNAMIC):

    # Variable to store sector values
    val = np.zeros(self.nlayer - 1)

    # Reshape fo state according to NN dimensions
    x = x.reshape(1, self.W[0].shape[1])
    
    # Initialization of empty vector to store events
    e = np.zeros(self.nlayer - 1)

    # Loop for each layer
    for l in range(self.nlayer - 1):
      
      # If we are in the fist layer the input is initial x, otherwise it is the omega of the previous layer
      if l == 0:
        # Detach, numpy and reshape are used to cast to np array of the correct shape in order to hav
        nu = self.layers[l](torch.tensor(x)).detach().numpy().reshape(self.W[l].shape[0], 1)
      else:
        nu = self.layers[l](torch.tensor(omega.reshape(1, self.W[l].shape[1]))).detach().numpy().reshape(self.W[l].shape[0], 1)

      # ETM evaluation
      vec1 = (nu - self.last_w[l]).T
      T = self.T[l]
      vec2 = (self.G[l] @ (x - self.xstar).reshape(self.nx, 1) - (self.last_w[l] - self.wstar[l]))


      # Flag to enbale/disable ETM
      if ETM:
        if DYNAMIC:
          check = vec1 @ T @ vec2 > self.rho[l] * self.eta[l]
        else:
          check = vec1 @ T @ vec2 > 0
      else:
        check = True

      if check:
        # If there is an event we compute the output of the layer with the non linear activation function
        omega = self.saturation_activation(torch.tensor(nu))
        # We substitute last computed value
        self.last_w[l] = omega.detach().numpy()
        # I flag the event at layer l
        e[l] = 1
        # I take the new omega to compute again the sector conditons that now are negative
        omega = omega.detach().numpy()
        vec1 = (nu - omega).T
        T = self.T[l]
        vec2 = (self.G[l] @ (x - self.xstar).reshape(self.nx, 1) - (omega - self.wstar[l]))
        val[l] = (vec1 @ T @ vec2)[0][0]
        
      else:
        val[l] = (vec1 @ T @ vec2)[0][0]
        # If no event occurs we feed the next layer the last stored output
        omega = self.last_w[l]

    # Last layer, different since it doesn't need ETM evaluation and w value update
    l = self.nlayer - 1
    nu = self.layers[l](torch.tensor(omega.reshape(1, self.W[l].shape[1])))
    omega = nu.detach().numpy().reshape(self.W[l].shape[0], 1)

    # Eta dyamics
    for i in range(self.nlayer - 1):
      self.eta[i] = (self.rho[i] + self.lam[i]) * self.eta[i] - val[i]

    return omega, e, self.eta

  ## Customly defined activation function since sat doesn't exist on tensorflow
  def saturation_activation(self, value):
    return torch.clamp(value, min=-self.bound, max=self.bound)

  ## Function that evaluates the closed loop dynamics
  def dynamic_loop(self, x0, nstep, ETM, DYNAMIC):

    # Initializes the state variable of the system to the initial condition
    if self.state is None:
      self.state = x0

    states = []
    inputs = []
    events = []
    etas1 = []
    etas2 = []

    # Loop called at each step
    for i in range(nstep):

      # Input computation via NN controller, events are tracked
      u, e, eta = self.forward(self.state, ETM, DYNAMIC)
      
      # Forward dynamics
      x = (self.A + self.B @ self.K) @ self.state.reshape(2,1) + self.B @ u.reshape(1, 1)

      # Optional noise addition
      # self.state = x + (np.random.randn(self.nx)*self.stdx).reshape(2, 1)
      self.state = x
      # u = u + np.random.randn(self.nu)*self.stdu

      states.append(x)
      inputs.append(u)
      events.append(e)
      etas1.append(eta[0])
      etas2.append(eta[1])

    return np.array(states), np.array(inputs), np.array(events), np.array(etas1), np.array(etas2)
  

# Main execution
if __name__ == "__main__":

  # Systam object creation
  s = System()

  # Import of P matrix from LMI solution
  P_mat_name = os.path.abspath(__file__ + "/../mat-weights/P_try.npy")
  P = np.load(P_mat_name)
  
  # Empty vectors initialization
  states = []
  inputs = []
  events = []
  etas1 = []
  etas2 = []
  lyap = []

  # Simulation parameters
  x0 = np.array([np.pi/6, 0.1])
  # In time it's nstep*s.dt = nstep * 0.02 s
  if len(sys.argv) > 1:
    nstep = int(sys.argv[1])
    if (int(sys.argv[2])):
      ETM = True
    else:
      ETM = False
    if (int(sys.argv[3])):
      DYNAMIC = True
    else:
      DYNAMIC = False
    if (int(sys.argv[4])):
      print_events = True
    else:
      print_events = False
  else:
    nstep = 250
    ETM = True
    DYNAMIC = True
    print_events = True

  # Call to simulation function of object System
  states, inputs, events, etas1, etas2 = s.dynamic_loop(x0, nstep, ETM, DYNAMIC)

  layer1_trigger = np.sum(events[:, 0]) / nstep * 100
  layer2_trigger = np.sum(events[:, 1]) / nstep * 100

  print("Layer 1 has been triggered " + str(layer1_trigger) + "% of times")
  print("Layer 2 has been triggered " + str(layer2_trigger) + "% of times")
  
  # Unpacking vectors
  x = states[:, 0]
  v = states[:, 1]
  u = inputs.reshape(len(inputs))
  time_grid = np.linspace(0, nstep * s.dt, nstep)
  
  # Creation of lyapunov function vector
  for i in range(nstep):
    lyap.append(((states[i, :] - s.xstar).T @ P @ (states[i, :] - s.xstar))[0][0] + 2*etas1[i] + 2*etas2[i])

  # Conversion of non events to None for ease of plotting
  for i, event in enumerate(events):
    if not event[0]:
      events[i][0] = None
    if not event[1]:
      events[i][1] = None

  ## Plotting
  
  fig, axs = plt.subplots(2, 2)
  axs[0, 0].plot(time_grid, x - s.xstar[0])
  if print_events:
    axs[0, 0].plot(time_grid, events[:, 1]*(x - s.xstar[0]).reshape(len(time_grid)), marker='o', markerfacecolor='none')
  axs[0, 0].set_title("Position")
  axs[0, 0].set_xlabel("Time")
  axs[0, 0].set_ylabel("Position [rad]")
  axs[0, 0].grid(True)

  axs[0, 1].plot(time_grid, v - s.xstar[1])
  if print_events:
    axs[0, 1].plot(time_grid, events[:, 1]*(v - s.xstar[1]).reshape(len(time_grid)), marker='o', markerfacecolor='none')
  axs[0, 1].set_title("Velocity")
  axs[0, 1].set_xlabel("Time")
  axs[0, 1].set_ylabel("Velocity")
  axs[0, 1].grid(True)

  axs[1, 0].plot(time_grid, u)
  if print_events:
    axs[1, 0].plot(time_grid, events[:, 1]*u, marker='o', markerfacecolor='none')
  axs[1, 0].set_title("Inputs")
  axs[1, 0].set_xlabel("Time")
  axs[1, 0].set_ylabel("Control input")
  axs[1, 0].grid(True)

  axs[1, 1].plot(time_grid, lyap)
  if print_events:
    axs[1, 1].plot(time_grid, events[:, 1]*lyap, marker='o', markerfacecolor='none')
  axs[1, 1].set_title("Lyapunov function")
  axs[1, 1].grid(True)

  plt.show()

  x1_vals = []
  x2_vals = []
  x11_vals = []
  x21_vals = []

  P0 = np.array([[0.2916, 0.0054], [0.0054, 0.0090]])

  for i in range(100000):
    x1 = np.random.uniform(-10, 10) - s.xstar[0]
    x2 = np.random.uniform(-20, 20) - s.xstar[1]
    vec = np.array([x1, x2])
    if (vec.T @ P @ vec < 1):
      x1_vals.append(x1)
      x2_vals.append(x2)
    if (vec.T @ P0 @ vec < 1):
      x11_vals.append(x1)
      x21_vals.append(x2)

  plt.plot(x1_vals, x2_vals)
  plt.plot(x11_vals, x21_vals)
  plt.plot(x, v)
  plt.xlabel('x1')
  plt.ylabel('x2')
  plt.title('ROA')
  plt.grid(True)
  plt.legend()
  plt.show()

  plt.plot(time_grid, etas1)
  plt.plot(time_grid, etas2)
  plt.show()