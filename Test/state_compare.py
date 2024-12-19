import numpy as np
import torch.nn as nn
import torch
from scipy.linalg import block_diag
from scipy.optimize import fsolve
import warnings

class System():
  def __init__(self, W, b, Omega, ref, path):

    # Ignore useless warnings from torch
    warnings.filterwarnings("ignore", category=UserWarning)

    # Directory where the LMI parameters are stored
    self.path = path

    # State of the system variable 
    self.state = None

    # Constants
    self.g = 9.81
    self.m = 0.15
    self.l = 0.5
    self.mu = 0.05
    self.dt = 0.02
    self.max_torque = 5
    self.max_speed = 8.0
    self.constant_reference = ref
    
    # Dimensions of state, input of the system and non-linearity
    self.nx = 3
    self.nu = 1
    self.nq = 1

    # State matrices in the form x⁺ = Ax + Bu + C phi * D ref
    self.A = np.array([
        [1,                       self.dt,                                0],
        [self.g*self.dt/self.l,   1-self.mu*self.dt/(self.m*self.l**2),   0],
        [1,                       0,                                      1]
    ])
    self.B = np.array([
        [0],
        [self.dt * self.max_torque/(self.m*self.l**2)],
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

    ## NN-related variables
    self.nlayers = 4 # Considering the final saturation of the input
    self.neurons = [32, 32, 32, 1]

    # Weights and biases
    self.W = W
    self.b = b

    # List of layers of the neural network
    self.layers = []
    for i in range(self.nlayers):
      layer = nn.Linear(W[i].shape[1], W[i].shape[0])
      layer.weight = nn.Parameter(torch.tensor(W[i]))
      layer.bias = nn.Parameter(torch.tensor(b[i]))
      self.layers.append(layer)
    
    # Total number of neurons and activation functions
    self.nphi = self.W[0].shape[0] + self.W[1].shape[0] + self.W[2].shape[0] + self.W[3].shape[0]

    # Saturation bound
    self.bound = 1

    # NN matrices in the form [u, v] = N [x, w, 1]
    N = block_diag(*self.W)
    self.Nux = np.zeros((self.nu, self.nx))
    self.Nuw = np.concatenate([np.zeros((self.nu, self.nphi - 1)), np.eye(self.nu)], axis=1)
    self.Nub = np.array([[0.0]])
    self.Nvx = N[:, :self.nx]
    self.Nvw = np.concatenate([N[:, self.nx:], np.zeros((self.nphi, self.nu))], axis=1)
    self.Nvb = np.concatenate([b_i.reshape(-1, 1) for b_i in self.b], axis=0)

    self.N = [self.Nux, self.Nuw, self.Nub, self.Nvx, self.Nvw, self.Nvb]

    # Useful matrices for LMI and equilibria computation
    self.R = np.linalg.inv(np.eye(*self.Nvw.shape) - self.Nvw)
    self.Rw = self.Nux + self.Nuw @ self.R @ self.Nvx
    self.Rb = self.Nuw @ self.R @ self.Nvb + self.Nub

    # Equilibrium computation with implicit form
    def implicit_function(x):
      x = x.reshape(3, 1)
      I = np.eye(self.A.shape[0])
      K = np.array([[1.0, 0.0, 0.0]])
      to_zero = np.squeeze((-I + self.A + self.B @ self.Rw - self.C @ K) @ x + self.C * np.sin(K @ x) + self.D * self.constant_reference + self.B @ self.Rb)
      return to_zero

    self.xstar = fsolve(implicit_function, np.array([[self.constant_reference], [0.0], [0.0]])).reshape(3,1)

    # Equilibrium of the input value
    self.ustar = self.Rw @ self.xstar + self.Rb

    # Equilibrium state for hidden layers
    wstar = self.R @ self.Nvx @ self.xstar + self.R @ self.Nvb
    wstar1 = wstar[:self.neurons[0]]
    wstar2 = wstar[self.neurons[0]:self.neurons[0] + self.neurons[1]]
    wstar3 = wstar[self.neurons[0] + self.neurons[1]:self.neurons[0] + self.neurons[1] + self.neurons[2]]
    wstar4 = wstar[-1]
    self.wstar = [wstar1, wstar2, wstar3, wstar4]

    # ETM related parameters
    self.eta = np.ones(self.nlayers) * 0.0
    self.rho = np.ones(self.nlayers) * np.load(self.path + '/Rho.npy')[0][0]

    # Last output of the neural network for each layer, initialized to arbitrary high value to trigger an event on initialization
    self.last_w = []
    for i, neuron in enumerate(self.neurons):
      self.last_w.append(np.ones((neuron, 1))*1e8)

    # List to store ETM triggering matrices
    self.bigX = Omega
  
  # Function to compute the forward pass of the neural network taking into account all ETMs
  def forward(self):
    # Activation function definition to be called upon event
    func = nn.Hardtanh()

    # Empty vector to keep track of events
    e = np.zeros(self.nlayers)

    # Reshape state for input
    x = self.state.reshape(1, self.W[0].shape[1])

    # Empty vector to store the values needed for the dynamic of the dynamic ETM thresholds
    val = np.zeros(self.nlayers)

    # Iteration for each layer
    for l in range(self.nlayers):

      # Particular case for the first layer since the input is the state
      if l == 0:
        input = torch.tensor(x)
      else:
        if omega is None:
        # fake omega for code robustness
          omega = np.zeros((1, self.W[l].shape[1]))
        
        # The input is the output of the previous layer
        input = torch.tensor(omega.reshape(1, self.W[l].shape[1]))
      
      # Forward pass without activation function
      nu = self.layers[l](input).detach().numpy().reshape(self.W[l].shape[0], 1)

      # Event computation: gamma * Psi >= rho * eta
      # Right hand term
      rht = self.rho[l] * self.eta[l]
      # Left hand term: gamma * [xtilde, psitilde, nutilde]^T @ X @ [xtilde, psitilde, nutilde]
      xtilde = self.state.reshape(3,1) - self.xstar.reshape(3, 1)
      psitilde = nu - self.last_w[l]
      nutilde = nu - self.wstar[l]
      vec = np.vstack([xtilde, psitilde, nutilde])
      mat = self.bigX[l]
      lht = (vec.T @ mat @ vec)[0][0]

      event = lht > rht

      # If event is triggered, update the hidden layer values with the activation function and store the new value as the last output of the layer. Store the lht value as it will be needed by the eta dynamics
      if event:
        omega = func(torch.tensor(nu)).detach().numpy()
        self.last_w[l] = omega
        e[l] = 1
        psitilde = nu - omega
        vec = np.vstack([xtilde, psitilde, nutilde])
        lht = (vec.T @ mat @ vec)[0][0]
        val[l] = lht
      
      # If no event is triggered, store the last output of the layer as the current output
      else:
        val[l] = lht
        omega = self.last_w[l]
      
    # Eta dynamics
    for i in range(self.nlayers):
      self.eta[i] = self.rho[i] * self.eta[i] - val[i]
    
    # Returns the last output value and the event vector
    return omega, e
  
  # Function to compute the state evolution of the system
  def step(self):
    # Input computation
    u, e = self.forward()
    # Non linearity computation
    nonlin = np.sin(self.state[0]) - self.state[0]
    # State evolution
    self.state = self.A @ self.state + self.B @ u + self.C * nonlin + self.D * self.constant_reference
    # Eta vales extraction
    etaval = self.eta.tolist()
    return self.state, u, e, etaval

if __name__ == "__main__":
  import os

  path1 = 'finstatic'
  # path2 = 'finsler099'
  # path2 = 'finlserrho05'
  path2 = 'finslerrho06'
  # path2 = 'finslerrho045'
  # path2 = 'optim'

  # Weights and biases import
  W1_name = os.path.abspath(__file__ + "/../weights/W1.csv")
  W2_name = os.path.abspath(__file__ + "/../weights/W2.csv")
  W3_name = os.path.abspath(__file__ + "/../weights/W3.csv")
  W4_name = os.path.abspath(__file__ + "/../weights/W4.csv")

  b1_name = os.path.abspath(__file__ + "/../weights/b1.csv")
  b2_name = os.path.abspath(__file__ + "/../weights/b2.csv")
  b3_name = os.path.abspath(__file__ + "/../weights/b3.csv")
  b4_name = os.path.abspath(__file__ + "/../weights/b4.csv")
  
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

  # New ETM matrices import system 1
  bigX1 = np.load(path1 + '/bigX1.npy')
  bigX2 = np.load(path1 + '/bigX2.npy')
  bigX3 = np.load(path1 + '/bigX3.npy')
  bigX4 = np.load(path1 + '/bigX4.npy')

  ETM1 = [bigX1, bigX2, bigX3, bigX4]

  # New ETM matrices import system 2
  bigX1 = np.load(path2 + '/bigX1.npy')
  bigX2 = np.load(path2 + '/bigX2.npy')
  bigX3 = np.load(path2 + '/bigX3.npy')
  bigX4 = np.load(path2 + '/bigX4.npy')

  ETM2 = [bigX1, bigX2, bigX3, bigX4]

  # System initialization
  s1 = System(W, b, ETM1, 0.0, path1)
  s2 = System(W, b, ETM2, 0.0, path2)

  P1 = np.load(path1 + '/P.npy')
  P2 = np.load(path1 + '/P.npy')
  
  # Maximum disturbance bound on the position theta in degrees
  ref_bound = 5 * np.pi / 180

  # Flag to check if the initial state is inside the ellipsoid
  in_ellip = False

  while not in_ellip:
    # Random initial state and disturbance
    theta = np.random.uniform(-np.pi/2, np.pi/2)
    vtheta = np.random.uniform(-s1.max_speed, s1.max_speed)
    ref = np.random.uniform(-ref_bound, ref_bound)

    # Initial state definition and system initialization
    x0 = np.array([[theta], [vtheta], [0.0]])
    s1 = System(W, b, ETM1, ref, path1)
    s2 = System(W, b, ETM2, ref, path2)

    # Check if the initial state is inside the ellipsoid
    if (x0 - s1.xstar).T @ P1 @ (x0 - s1.xstar) <= 1.0 and (x0 - s1.xstar).T @ P1 @ (x0 - s1.xstar) >= 0.9:

      # Initial eta0 computation with respect to the initial state
      eta02 = ((1 - (x0 - s2.xstar).T @ P2 @ (x0 - s2.xstar)) / (s2.nlayers * 2))[0][0]
      
      # Initial eta0 value update in the system
      s2.eta = np.ones(s2.nlayers) * eta02
      s2.rho = np.ones(s2.nlayers) * 0.9
      
      # Initial state update in the system
      s1.state = x0
      s2.state = x0

      # Flag variable update to stop the search
      in_ellip = True

  # Simulation loop
  # Empty lists to store the values of the simulation
  states1 = []
  states2 = []
  events1 = []
  events2 = []

  # Maximum number of steps to stop the simulation
  max_steps = 300

  # Simulation loop
  for i in range(max_steps):

    # Step computation
    state1, _, e1, _ = s1.step()
    state2, _, e2, _ = s2.step()

    # Values storage
    states1.append(state1)
    states2.append(state2)
    events1.append(e1)
    events2.append(e2)

  
  # Initial state added manually to the states list
  states1 = np.insert(states1, 0, x0, axis=0)
  states2 = np.insert(states2, 0, x0, axis=0)
  # Last state removed to have the same size as the other lists
  states1 = np.delete(states1, -1, axis=0)
  states2 = np.delete(states2, -1, axis=0)
  # First state component converted to degrees
  states1 = np.squeeze(np.array(states1))
  states2 = np.squeeze(np.array(states2))
  states1[:, 0] *= 180 / np.pi
  states2[:, 0] *= 180 / np.pi
  s1.xstar[0] *= 180 / np.pi
  s2.xstar[0] *= 180 / np.pi


  events1 = np.squeeze(np.array(events1))
  events2 = np.squeeze(np.array(events2))

  # Data visualization
  timegrid = np.arange(0, max_steps)

  # Triggering percentage computation
  layer1_trigger1 = np.sum(events1[:, 0]) / max_steps * 100
  layer2_trigger1 = np.sum(events1[:, 1]) / max_steps * 100
  layer3_trigger1 = np.sum(events1[:, 2]) / max_steps * 100
  layer4_trigger1 = np.sum(events1[:, 3]) / max_steps * 100
  overall1 = (layer1_trigger1 * s1.neurons[0] + layer2_trigger1 * s1.neurons[1] + layer3_trigger1 * s1.neurons[2] + layer4_trigger1 * s1.neurons[3]) / (s1.nphi)

  layer1_trigger2 = np.sum(events2[:, 0]) / max_steps * 100
  layer2_trigger2 = np.sum(events2[:, 1]) / max_steps * 100
  layer3_trigger2 = np.sum(events2[:, 2]) / max_steps * 100
  layer4_trigger2 = np.sum(events2[:, 3]) / max_steps * 100
  overall2 = (layer1_trigger2 * s2.neurons[0] + layer2_trigger2 * s2.neurons[1] + layer3_trigger2 * s2.neurons[2] + layer4_trigger2 * s2.neurons[3]) / (s2.nphi)
  
  print(f"Static: {layer1_trigger1:.2f}, {layer2_trigger1:.2f}, {layer3_trigger1:.2f}, {layer4_trigger1:.2f}, tot: {overall1:.2f} ")
  print(f"Dynamic: {layer1_trigger2:.2f}, {layer2_trigger2:.2f}, {layer3_trigger2:.2f}, {layer4_trigger2:.2f}, tot: {overall2:.2f} ")

  # Replace every non event value from 0 to None for ease of plotting
  for i, event in enumerate(events1):
    if not event[0]:
      events1[i][0] = None
    if not event[1]:
      events1[i][1] = None
    if not event[2]:
      events1[i][2] = None
    if not event[3]:
      events1[i][3] = None
  
  for i, event in enumerate(events2):
    if not event[0]:
      events2[i][0] = None
    if not event[1]:
      events2[i][1] = None
    if not event[2]:
      events2[i][2] = None
    if not event[3]:
      events2[i][3] = None
      
  import matplotlib.pyplot as plt

  # Control input plot
  plot_cut = 5000
  fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

  # Theta plot
  axs[0].plot(timegrid[:plot_cut], states1[:plot_cut, 0], label=r'$\theta$')
  # Theta star plot
  axs[0].plot(timegrid[:plot_cut], timegrid[:plot_cut] * 0 + s1.xstar[0], 'r--', label=r'$\theta_*$')
  axs[0].plot(timegrid[:plot_cut], states1[:plot_cut, 0] * events1[:plot_cut, 3], marker='o', markerfacecolor='none', linestyle='None', label='Events')
  # axs[1].set_xlabel('Time steps',fontsize=14)
  axs[0].set_ylabel(r'$\theta$ (deg)',fontsize=14)
  axs[0].legend(fontsize=14, loc='upper right', ncols=3)
  axs[0].grid(True)

  axs[1].plot(timegrid[:plot_cut], states2[:plot_cut, 0], label=r'$\theta$')
  # Theta star plot
  axs[1].plot(timegrid[:plot_cut], timegrid[:plot_cut] * 0 + s2.xstar[0], 'r--', label=r'$\theta_*$')
  axs[1].plot(timegrid[:plot_cut], states2[:plot_cut, 0] * events2[:plot_cut, 3], marker='o', markerfacecolor='none', linestyle='None', label='Events')
  # axs[1].set_xlabel('Time steps',fontsize=14)
  axs[1].set_ylabel(r'$\theta$ (deg)',fontsize=14)
  axs[1].legend(fontsize=14, loc='upper right', ncols=3)
  axs[1].grid(True)
  plt.show()