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
    W5_name = os.path.abspath(__file__ + "/../4_layers/W5.csv")
    W5 = np.loadtxt(W5_name, delimiter=',')
    W5 = W5.reshape(self.nu, len(W5))

    # Vector of weigts
    self.W = [W1, W2, W3, W4, W5]

    # Number of neurons
    self.nphi = W1.shape[0] + W2.shape[0] + W3.shape[0] + W4.shape[0]
    
    # Bias import of trained FFNN
    b1_name = os.path.abspath(__file__ + "/../4_layers/b1.csv")
    b1 = np.loadtxt(b1_name, delimiter=',')
    b2_name = os.path.abspath(__file__ + "/../4_layers/b2.csv")
    b2 = np.loadtxt(b2_name, delimiter=',')
    b3_name = os.path.abspath(__file__ + "/../4_layers/b3.csv")
    b3 = np.loadtxt(b3_name, delimiter=',')
    b4_name = os.path.abspath(__file__ + "/../4_layers/b4.csv")
    b4 = np.loadtxt(b4_name, delimiter=',')
    b5_name = os.path.abspath(__file__ + "/../4_layers/b5.csv")
    b5 = np.loadtxt(b5_name, delimiter=',')

    # Vector of biases
    self.b = [b1, b2, b3, b4, b5]

    self.nlayer = len(self.W)

    # Neurons per layer in the FFNN (to substitute with array if different)
    neurons = 16*np.ones((1, self.nlayer - 1)).astype(np.int16)
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

    T = np.load('./4_layers/T_mat.npy')
    Z = np.load('./4_layers/Z_mat.npy')
    G = np.linalg.inv(T) @ Z
    self.G = np.split(G, self.nlayer - 1)
    self.block_T = T
    self.T = []
    for i in range(self.nlayer - 1):
      self.T.append(T[i*self.neurons[i]:(i+1)*self.neurons[i], i*self.neurons[i]:(i+1)*self.neurons[i]])

    # Closed loop matrices
    N = block_diag(*self.W) # Reminder for myself: * operator unpacks lists and pass it as singular arguments
    Nux = self.K
    Nuw = N[-self.nu:, self.nx:]
    Nub = self.b[-1].reshape(self.nu, self.nu)
    Nvx = N[:-self.nu, :self.nx]
    Nvw = N[:-self.nu, self.nx:]
    Nvb = np.array([[self.b[0]], [self.b[1]], [self.b[2]], [self.b[3]]]).reshape(self.nphi, self.nu)

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

    self.last_w = []
    for i, neuron in enumerate(self.neurons):
      self.last_w.append(np.ones((neuron, 1))*1e3)

    ## Class variable
    self.state = None

    self.eta = 1
    self.rho = 0.8
    self.lam = 0.1

    self.val = []

    ## Function that predicts the input
  def forward(self, x):

    val = 0
    e = np.zeros(self.nlayer - 1)

    # Reshape fo state according to NN dimensions
    x = x.reshape(1, self.W[0].shape[1])

    for l in range(self.nlayer - 1):
      if l == 0:
        input = torch.tensor(x)
      else:
        input = torch.tensor(omega.reshape(1, self.W[l].shape[1]))

      nu = self.layers[l](input).detach().numpy().reshape(self.W[l].shape[0], 1)

      # ETM evaluation
      vec1 = (nu - self.last_w[l]).T
      T = self.T[l]
      vec2 = (self.G[l] @ (x.reshape(self.nx, 1) - self.xstar) - (self.last_w[l] - self.wstar[l]))
      
      print(f"ETM CONDITION: {(vec1 @ T @ vec2)[0][0]:.2f}  ETA VALUE: {(self.rho * self.eta):.2f}")

      check = vec1 @ T @ vec2 > 0

      if check:
        omega = self.saturation_activation(torch.tensor(nu))
        omega = omega.detach().numpy()
        self.last_w[l] = omega
        e[l] = 1
        vec1 = (nu - omega).T   
        T = self.T[l]
        vec2 = (self.G[l] @ (x.reshape(self.nx, 1) - self.xstar) - (omega - self.wstar[l]))
        val += (vec1 @ T @ vec2)[0][0]
        self.val.append((vec1 @ T @ vec2)[0][0])
      else:
        val += (vec1 @ T @ vec2)[0][0]
        omega = self.last_w[l]
    
    # Last layer
    l = self.nlayer - 1
    omega = self.layers[l](torch.tensor(omega.reshape(1, self.W[l].shape[1]))).detach().numpy().reshape(self.W[l].shape[0], 1)

    # Eta dynamics
    self.eta = (self.rho + self.lam) * self.eta - val

    return omega, e, self.eta

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
    events = []
    etas = []

    # Loop called at each step
    for i in range(nstep):

      # Input computation via NN controller, events are tracked
      u, e, eta = self.forward(self.state)
      
      # Forward dynamics
      x = (self.A + self.B @ self.K) @ self.state.reshape(2,1) + self.B @ u.reshape(1, 1)

      # Optional noise addition
      # self.state = x + (np.random.randn(self.nx)*self.stdx).reshape(2, 1)
      self.state = x
      # u = u + np.random.randn(self.nu)*self.stdu

      states.append(x)
      inputs.append(u)
      events.append(e)
      etas.append(eta)

    return np.array(states), np.array(inputs), np.array(events), np.array(etas)
  

# Main execution
if __name__ == "__main__":

  # Systam object creation
  s = System()

  check_lyap = True

  P = np.load('./4_layers/P0_mat.npy')*10

  n_test = 1

  for i in range(n_test):
    
    # Empty vectors initialization
    states = []
    inputs = []
    events = []
    etas = []
    lyap = []

    x1 = np.pi/4
    x2 = 0.1
    x0 = np.array([x1, x2])
    x1_deg = x1/np.pi*180
    print(f"Initial state: position: {x1_deg:.2f}°, speed: {x2:.2f} [rad/s]")

    nstep = 300
    # Call to simulation function of object System
    s.state = None
    states, inputs, events, etas = s.dynamic_loop(x0, nstep)

    # Unpacking vectors
    x = states[:, 0]
    v = states[:, 1]
    u = inputs.reshape(len(inputs))
    time_grid = np.linspace(0, nstep * s.dt, nstep)

    if check_lyap:
      for i in range(nstep):
        lyap.append((states[i] - s.xstar).T @ P @ (states[i] - s.xstar) + etas[i])
      lyap = np.array(lyap).reshape(nstep)
    
    for i, event in enumerate(events):
      if not event[0]:
        events[i][0] = None
      if not event[1]:
        events[i][1] = None
    
    ## Plotting
    
    fig, axs = plt.subplots(5)

    axs[0].plot(time_grid, x - s.xstar[0])
    axs[0].plot(time_grid, events[:, 1]*(x - s.xstar[0]).reshape(len(time_grid)), marker='o', markerfacecolor='none')
    axs[0].set_ylabel('x1')
    axs[0].grid(True)

    axs[1].plot(time_grid, v - s.xstar[1])
    axs[1].plot(time_grid, events[:, 1]*(v - s.xstar[1]).reshape(len(time_grid)), marker='o', markerfacecolor='none')
    axs[1].set_ylabel('x2')
    axs[1].grid(True)

    axs[2].plot(time_grid, u)
    axs[2].plot(time_grid, events[:, 1]*u.reshape(len(time_grid)), marker='o', markerfacecolor='none')
    axs[2].set_ylabel('u')
    axs[2].grid(True)

    axs[3].plot(time_grid, lyap)
    axs[3].set_ylabel('lyap')
    axs[3].grid(True)

    axs[4].plot(time_grid, etas)
    axs[4].set_ylabel('eta')
    axs[4].grid(True)

    plt.show()

