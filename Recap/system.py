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
    self.rho = np.ones(self.nlayers) * np.load(self.path + '/Rho.npy')[0][0]
    self.lambda1 = np.load(self.path + '/lambda1.npy')
    self.gamma_low = np.load(self.path + '/gamma_low.npy')
    self.gamma_high = np.load(self.path + '/gamma_high.npy')
    self.gammas = np.ones(self.nlayers) * (self.gamma_low + self.lambda1 * (self.gamma_high - self.gamma_low))

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
      lht = self.gammas[l] * (vec.T @ mat @ vec)[0][0]

      event = lht > rht

      # If event is triggered, update the hidden layer values with the activation function and store the new value as the last output of the layer. Store the lht value as it will be needed by the eta dynamics
      if event:
        omega = func(torch.tensor(nu)).detach().numpy()
        self.last_w[l] = omega
        e[l] = 1
        psitilde = nu - omega
        vec = np.vstack([xtilde, psitilde, nutilde])
        lht = self.gammas[l] * (vec.T @ mat @ vec)[0][0]
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

  # path = 'finsler-0.5'
  path = 'finsler-1'
  # path = 'finsler-0'

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

  # Old ETM matrices import
  Omega1 = np.load('omega_dynamic/Omega1.npy')
  Omega2 = np.load('omega_dynamic/Omega2.npy')
  Omega3 = np.load('omega_dynamic/Omega3.npy')
  Omegas = np.load('omega_dynamic/Omegas.npy')

  Omega = [Omega1, Omega2, Omega3, Omegas]

  # New ETM matrices import
  bigX1 = np.load(path + '/bigX1.npy')
  bigX2 = np.load(path + '/bigX2.npy')
  bigX3 = np.load(path + '/bigX3.npy')
  bigX4 = np.load(path + '/bigX4.npy')

  bigX = [bigX1, bigX2, bigX3, bigX4]

  # System initialization
  s = System(W, b, bigX, 0.0, path)
  
  # P matrix import for lyapunov function
  P = np.load(path + '/P.npy')

  print(f"Size of ROA: {np.pi/np.sqrt(np.linalg.det(P)):.2f}")

  # Maximum disturbance bound on the position theta in degrees
  ref_bound = 5 * np.pi / 180

  # Flag to check if the initial state is inside the ellipsoid
  in_ellip = False

  # Loop to find a random initial state inside the ellipsoid
  while not in_ellip:
    # Random initial state and disturbance
    theta = np.random.uniform(-np.pi/2, np.pi/2)
    vtheta = np.random.uniform(-s.max_speed, s.max_speed)
    ref = np.random.uniform(-ref_bound, ref_bound)

    theta = 5.0 * np.pi / 180
    vtheta = -0.5
    ref = 1 * np.pi / 180

    # Initial state definition and system initialization
    x0 = np.array([[theta], [vtheta], [0.0]])
    s = System(W, b, bigX, ref, path)

    # Check if the initial state is inside the ellipsoid
    if (x0 - s.xstar).T @ P @ (x0 - s.xstar) <= 1.0: # and (x0).T @ P @ (x0) >= 0.9:
      # Initial eta0 computation with respect to the initial state
      eta0 = ((1 - (x0 - s.xstar).T @ P @ (x0 - s.xstar)) / (s.nlayers * 2))[0][0]*0
      
      # Flag variable update to stop the search
      in_ellip = True
      
      # Initial eta0 value update in the system
      s.eta = np.ones(s.nlayers) * eta0
      
      # Initial state update in the system
      s.state = x0

      print(f"Initial state: theta0 = {theta*180/np.pi:.2f} deg, vtheta0 = {vtheta:.2f} rad/s, constant reference = {ref*180/np.pi:.2f} deg")
      print(f"Initial eta0: {eta0:.2f}")

  # Simulation loop
  
  # Empty lists to store the values of the simulation
  states = []
  inputs = []
  events = []
  etas = []
  lyap = []

  # Flag to stop the simulation
  stop_run = False

  # Counter of the number of steps
  nsteps = 0

  # Magnitude of the Lyapunov function to stop the simulation
  lyap_magnitude = 1e-2

  # Simulation loop
  while not stop_run:
    # Counter update
    nsteps += 1

    # Step computation
    state, u, e, eta = s.step()

    # Values storage
    states.append(state)
    inputs.append(u)
    events.append(e)
    etas.append(eta)
    lyap.append((state - s.xstar).T @ P @ (state - s.xstar) + 2*eta[0] + 2*eta[1] + 2*eta[2] + 2*eta[3])
    
    # Stop condition
    if lyap[-1] < lyap_magnitude:
      stop_run = True

  # Data processing
  
  # Initial state added manually to the states list
  states = np.insert(states, 0, x0, axis=0)
  # Last state removed to have the same size as the other lists
  states = np.delete(states, -1, axis=0)
  # First state component converted to degrees
  states = np.squeeze(np.array(states))
  states[:, 0] *= 180 / np.pi
  s.xstar[0] *= 180 / np.pi

  # Initial input added manually to the inputs list, set to 0
  inputs = np.insert(inputs, 0, np.array(0.0), axis=0)
  # Last input removed to have the same size as the other lists
  inputs = np.delete(inputs, -1, axis=0)
  # Inputs multiplied by the maximum torque to have the real value
  inputs = np.squeeze(np.array(inputs)) * s.max_torque

  events = np.squeeze(np.array(events))
  etas = np.squeeze(np.array(etas))
  lyap = np.squeeze(np.array(lyap))

  # Check of decrement of the Lyapunov function
  lyap_diff = np.diff(lyap)
  if np.all(lyap_diff <= 1e-25):
    print("Lyapunov function is always decreasing.")
  else:
    print("Lyapunov function is not always decreasing.")

  # Data visualization
  timegrid = np.arange(0, nsteps)

  # Triggering percentage computation
  layer1_trigger = np.sum(events[:, 0]) / nsteps * 100
  layer2_trigger = np.sum(events[:, 1]) / nsteps * 100
  layer3_trigger = np.sum(events[:, 2]) / nsteps * 100
  layer4_trigger = np.sum(events[:, 3]) / nsteps * 100

  print(f"Layer 1 has been triggered {layer1_trigger:.1f}% of times")
  print(f"Layer 2 has been triggered {layer2_trigger:.1f}% of times")
  print(f"Layer 3 has been triggered {layer3_trigger:.1f}% of times")
  print(f"Output layer has been triggered {layer4_trigger:.1f}% of times")

  # Replace every non event value from 0 to None for ease of plotting
  for i, event in enumerate(events):
    if not event[0]:
      events[i][0] = None
    if not event[1]:
      events[i][1] = None
    if not event[2]:
      events[i][2] = None
    if not event[3]:
      events[i][3] = None
      
  import matplotlib.pyplot as plt

  # Control input plot
  fig, axs = plt.subplots(4, 1)
  axs[0].plot(timegrid, inputs, label='Control input')
  axs[0].plot(timegrid, inputs * events[:, 3], marker='o', markerfacecolor='none', linestyle='None')
  # Ustar plot
  axs[0].plot(timegrid, np.squeeze(timegrid * 0 + s.ustar * s.max_torque), 'r--')
  axs[0].set_xlabel('Time steps')
  axs[0].set_ylabel('Values')
  axs[0].legend()
  axs[0].grid(True)

  # Theta plot
  axs[1].plot(timegrid, states[:, 0], label='Position')
  axs[1].plot(timegrid, states[:, 0] * events[:, 3], marker='o', markerfacecolor='none', linestyle='None')
  # Theta star plot
  axs[1].plot(timegrid, timegrid * 0 + s.xstar[0], 'r--')
  axs[1].set_xlabel('Time steps')
  axs[1].set_ylabel('Values')
  axs[1].legend()
  axs[1].grid(True)

  # V plot
  axs[2].plot(timegrid, states[:, 1], label='Velocity')
  axs[2].plot(timegrid, states[:, 1] * events[:, 3], marker='o', markerfacecolor='none', linestyle='None')
  # V star plot
  axs[2].plot(timegrid, timegrid * 0 + s.xstar[1], 'r--')
  axs[2].set_xlabel('Time steps')
  axs[2].set_ylabel('Values')
  axs[2].legend()
  axs[2].grid(True)

  # Integrator state plot
  axs[3].plot(timegrid, states[:, 2], label='Integrator state')
  axs[3].plot(timegrid, states[:, 2] * events[:, 3], marker='o', markerfacecolor='none', linestyle='None')
  # Integrator state star plot
  axs[3].plot(timegrid, timegrid * 0 + s.xstar[2], 'r--')
  axs[3].set_xlabel('Time steps')
  axs[3].set_ylabel('Values')
  axs[3].legend()
  axs[3].grid(True)
  plt.show()

  # Eta plots
  plt.plot(timegrid, etas[:, 0], label='Eta_1')
  plt.plot(timegrid, etas[:, 1], label='Eta_2')
  plt.plot(timegrid, etas[:, 2], label='Eta_3')
  plt.plot(timegrid, etas[:, 3], label='Eta_4')
  plt.legend()
  plt.grid(True)
  plt.show()

  # Lyapunov function plot
  plt.plot(timegrid, lyap, label='Lyapunov function')
  plt.legend()
  plt.grid(True)
  plt.show()

  from systems_and_LMI.user_defined_functions.ellipsoid_plot_2D import ellipsoid_plot_2D
  from systems_and_LMI.user_defined_functions.ellipsoid_plot_3D import ellipsoid_plot_3D

  # 3D ROA plot
  fig, ax = ellipsoid_plot_3D(P, False, color='b', legend='ROA with dynamic ETM')
  ax.plot(states[:, 0] - s.xstar[0], states[:, 1] - s.xstar[1], states[:, 2] - s.xstar[2], 'b')
  ax.plot(0, 0, 0, marker='o', markersize=5, color='r')
  plt.legend()
  plt.show()

  # 2D projection on integrator state star plane of R
  fig, ax = ellipsoid_plot_2D(P[:2, :2], False, color='b', legend='ROA with dynamic ETM')
  ax.plot(states[:, 0] - s.xstar[0], states[:, 1]  - s.xstar[1], 'b')
  ax.plot(0, 0, marker='o', markersize=5, color='r')
  plt.legend()
  plt.show()