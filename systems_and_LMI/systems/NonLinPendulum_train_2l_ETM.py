from systems_and_LMI.systems.NonLinPendulum_train_2l import NonLinPendulum_train2l
import systems_and_LMI.systems.nonlin_2l_ETM.params as params
import numpy as np
import os
import torch.nn as nn
import torch
import warnings

# User warnings filter
warnings.filterwarnings("ignore", category=UserWarning, module='torch')

# New class definition that depends on the passed weights and biases
class NonLinPendulum_train2l_ETM(NonLinPendulum_train2l):
  
  def __init__(self, W, b, ref):
    super().__init__(W, b, ref)

    Z_name = os.path.abspath(__file__ + '/../nonlin_2l_ETM/Z.npy')
    T_name = os.path.abspath(__file__ + '/../nonlin_2l_ETM/T.npy')

    T = np.load(T_name)
    Z = np.load(Z_name)
    G = np.linalg.inv(T) @ Z
    self.G = np.split(G, [16, 32])
    self.Z = np.split(Z, [16, 32])
    self.T = []
    for i in range(self.nlayers - 1):
      self.T.append(T[i*self.neurons[i]:(i+1)*self.neurons[i], i*self.neurons[i]:(i+1)*self.neurons[i]])
    self.eta = np.ones(self.nlayers - 1)*params.eta0
    self.rho = params.rhos
    self.lam = params.lambdas
    self.last_w = []
    for i, neuron in enumerate(self.neurons):
      self.last_w.append(np.ones((neuron, 1))*1e3)
    self.last_w.append(np.array([[1e3]]))

  def forward(self):
    func = nn.Hardtanh()
    e = np.zeros(self.nlayers - 1)
    x = self.state.reshape(1, self.W[0].shape[1]) 
    val = np.zeros(self.nlayers - 1)

    for l in range(self.nlayers - 1):
      if l == 0:
        nu = self.layers[l](torch.tensor(x)).detach().numpy().reshape(self.W[l].shape[0], 1)
      else:
        nu = self.layers[l](torch.tensor(omega.reshape(1, self.W[l].shape[1]))).detach().numpy().reshape(self.W[l].shape[0], 1)
      
      xtilde = self.state - self.xstar
      psitilde = nu - self.last_w[l]
      nutilde = nu - self.wstar[l]
      vec = np.vstack((xtilde, psitilde, nutilde))

      mat = np.block([
        [np.zeros((self.nx, self.nx)), np.zeros((self.nx, self.neurons[l])), np.zeros((self.nx, self.neurons[l]))],
        [self.Z[l], self.T[l], -self.T[l]],
        [np.zeros((self.neurons[l], self.nx)), np.zeros((self.neurons[l], self.neurons[l])), np.zeros((self.neurons[l], self.neurons[l]))]
      ])


      lht = (vec.T @ mat @ vec)[0][0]
      rht = self.rho[l] * self.eta[l]

      check = lht  > rht
      
      if check:
        omega = func(torch.tensor(nu)).detach().numpy()
        self.last_w[l] = nu
        e[l] = 1
        psitilde = nu - omega
        vec = np.vstack((xtilde, psitilde, nutilde))
        val[l] = (vec.T @ mat @ vec)[0][0]
      
      else:
        val[l] = lht
        omega = self.last_w[l]
        
    l = self.nlayers - 1
    nu = func(self.layers[l](torch.tensor(omega.reshape(1, self.W[l].shape[1])))).detach().numpy()

    for i in range(self.nlayers - 1):
      self.eta[i] = (self.rho[i] + self.lam[i]) * self.eta[i] - val[i]
      
    return nu, e
  
  def step(self):
    u, e = self.forward()
    u = np.clip(u, -1.0, 1.0)*self.max_torque
    nonlin = np.sin(self.state[0]) - self.state[0]
    self.state = self.A @ self.state + self.B @ u + self.C * nonlin + self.D * self.constant_reference
    etaval = self.eta.tolist()
    return self.state, u, e, etaval

if __name__ == "__main__":
  import matplotlib.pyplot as plt
  from systems_and_LMI.user_defined_functions.ellipsoid_plot_2D import ellipsoid_plot_2D
  from systems_and_LMI.user_defined_functions.ellipsoid_plot_3D import ellipsoid_plot_3D

  W1_name = os.path.abspath(__file__ + "/../nonlin_2l_ETM/l1.weight.csv")
  W2_name = os.path.abspath(__file__ + "/../nonlin_2l_ETM/l2.weight.csv")
  W3_name = os.path.abspath(__file__ + "/../nonlin_2l_ETM/l3.weight.csv")
  
  b1_name = os.path.abspath(__file__ + "/../nonlin_2l_ETM/l1.bias.csv")
  b2_name = os.path.abspath(__file__ + "/../nonlin_2l_ETM/l2.bias.csv")
  b3_name = os.path.abspath(__file__ + "/../nonlin_2l_ETM/l3.bias.csv")

  W1 = np.loadtxt(W1_name, delimiter=',')
  W2 = np.loadtxt(W2_name, delimiter=',')
  W3 = np.loadtxt(W3_name, delimiter=',')
  W3 = W3.reshape((1, len(W3)))

  W = [W1, W2, W3]

  b1 = np.loadtxt(b1_name, delimiter=',')
  b2 = np.loadtxt(b2_name, delimiter=',')
  b3 = np.loadtxt(b3_name, delimiter=',')

  b = [b1, b2, b3]
  
  s = NonLinPendulum_train2l_ETM(W, b, 0.0)

  P = np.load(os.path.abspath(__file__ + "/../nonlin_2l_ETM/P.npy"))

  print(f"Size of ROA: {np.pi/np.sqrt(np.linalg.det(P)):.2f}")

  ref_bound = 5 * np.pi / 180
  in_ellip = False
  while not in_ellip:
    theta = np.random.uniform(-np.pi/2, np.pi/2)
    vtheta = np.random.uniform(-s.max_speed, s.max_speed)
    x0 = np.array([[theta], [vtheta], [0.0]])
    if (x0).T @ P @ (x0) <= 1.0:
      in_ellip = True
      ref = np.random.uniform(-ref_bound, ref_bound)
      s = NonLinPendulum_train2l_ETM(W, b, ref)
      print(f"Initial state: theta0 = {theta*180/np.pi:.2f} deg, vtheta0 = {vtheta:.2f} rad/s, constant reference = {ref*180/np.pi:.2f} deg")
      s.state = x0

  nsteps = 300

  states = []
  inputs = []
  events = []
  etas = []
  lyap = []

  for i in range(nsteps):
    state, u, e, eta = s.step()
    states.append(state)
    inputs.append(u)
    events.append(e)
    etas.append(eta)
    lyap.append((state - s.xstar).T @ P @ (state - s.xstar) + 2*eta[0] + 2*eta[1])

  states = np.insert(states, 0, x0, axis=0)
  states = np.delete(states, -1, axis=0)
  states = np.squeeze(np.array(states))
  states[:, 0] *= 180 / np.pi
  s.xstar[0] *= 180 / np.pi

  inputs = np.insert(inputs, 0, np.array(0.0), axis=0)
  inputs = np.delete(inputs, -1, axis=0)
  inputs = np.squeeze(np.array(inputs))

  events = np.squeeze(np.array(events))
  etas = np.squeeze(np.array(etas))
  lyap = np.squeeze(np.array(lyap))

  timegrid = np.arange(0, nsteps)

  layer1_trigger = np.sum(events[:, 0]) / nsteps * 100
  layer2_trigger = np.sum(events[:, 1]) / nsteps * 100

  print(f"Layer 1 has been triggered {layer1_trigger:.1f}% of times")
  print(f"Layer 2 has been triggered {layer2_trigger:.1f}% of times")

  for i, event in enumerate(events):
    if not event[0]:
      events[i][0] = None
    if not event[1]:
      events[i][1] = None
      
  fig, axs = plt.subplots(4, 1)
  axs[0].plot(timegrid, inputs, label='Control input')
  axs[0].plot(timegrid, inputs * events[:, 1], marker='o', markerfacecolor='none', linestyle='None')
  axs[0].plot(timegrid, timegrid * 0 + s.wstar[-1] * s.max_torque, 'r--')
  axs[0].set_xlabel('Time steps')
  axs[0].set_ylabel('Values')
  axs[0].legend()
  axs[0].grid(True)

  axs[1].plot(timegrid, states[:, 0], label='Position')
  axs[1].plot(timegrid, states[:, 0] * events[:, 1], marker='o', markerfacecolor='none', linestyle='None')
  axs[1].plot(timegrid, timegrid * 0 + s.xstar[0], 'r--')
  axs[1].set_xlabel('Time steps')
  axs[1].set_ylabel('Values')
  axs[1].legend()
  axs[1].grid(True)

  axs[2].plot(timegrid, states[:, 1], label='Velocity')
  axs[2].plot(timegrid, states[:, 1] * events[:, 1], marker='o', markerfacecolor='none', linestyle='None')
  axs[2].plot(timegrid, timegrid * 0 + s.xstar[1], 'r--')
  axs[2].set_xlabel('Time steps')
  axs[2].set_ylabel('Values')
  axs[2].legend()
  axs[2].grid(True)

  axs[3].plot(timegrid, states[:, 2], label='Integrator state')
  axs[3].plot(timegrid, states[:, 2] * events[:, 1], marker='o', markerfacecolor='none', linestyle='None')
  axs[3].plot(timegrid, timegrid * 0 + s.xstar[2], 'r--')
  axs[3].set_xlabel('Time steps')
  axs[3].set_ylabel('Values')
  axs[3].legend()
  axs[3].grid(True)
  plt.show()

  plt.plot(timegrid, etas[:, 0], label='Eta_1')
  plt.plot(timegrid, etas[:, 1], label='Eta_2')
  plt.legend()
  plt.grid(True)
  plt.show()

  plt.plot(timegrid, lyap, label='Lyapunov function')
  plt.legend()
  plt.grid(True)
  plt.show()

  fig, ax = ellipsoid_plot_3D(P, False, color='b', legend='ROA with dynamic ETM')
  ax.plot(states[:, 0], states[:, 1], states[:, 2], 'b')
  ax.plot(s.xstar[0], s.xstar[1], s.xstar[2], marker='o', markersize=5, color='r')
  plt.legend()
  plt.show()

  fig, ax = ellipsoid_plot_2D(P[:2, :2], False, color='b', legend='ROA with dynamic ETM')
  ax.plot(states[:, 0], states[:, 1], 'b')
  ax.plot(s.xstar[0], s.xstar[1], marker='o', markersize=5, color='r')
  plt.legend()
  plt.show()