from system import System
import os
import numpy as np
import torch.nn as nn
import torch
from systems_and_LMI.user_defined_functions.ellipsoid_plot_2D import ellipsoid_plot_2D
import params

xETM = True

class Better_System(System):
  
  def __init__(self):
    super().__init__()
    
    T_mat_name = os.path.abspath(__file__ + "/../mat-weights/T_try.npy")
    Z_mat_name = os.path.abspath(__file__ + "/../mat-weights/Z_try.npy")
    # T_mat_name = os.path.abspath(__file__ + "/../mat-weights/xT.npy")
    # Z_mat_name = os.path.abspath(__file__ + "/../mat-weights/xZ.npy")
    bigX1_name = os.path.abspath(__file__ + "/../Test/bigX1.npy")
    bigX2_name = os.path.abspath(__file__ + "/../Test/bigX2.npy")

    T = np.load(T_mat_name)
    Z = np.load(Z_mat_name)
    bigX1 = np.load(bigX1_name)
    bigX2 = np.load(bigX2_name)
    self.K = np.array([[-0.1, 0]])

    self.bigX = [bigX1, bigX2]

    self.Z = np.split(Z, [32, 64])
    self.T = []
    
    neurons = [32, 32]
    for i in range(2):
      self.T.append(T[i*neurons[i]:(i+1)*neurons[i], i*neurons[i]:(i+1)*neurons[i]])
    
    self.eta = np.zeros(2)
    self.rho = params.rhos
    self.lam = params.lambdas

  def forward(self):
    func = nn.Hardtanh()
    e = np.zeros(2)
    x = self.state.reshape(1, self.W[0].shape[1])
    val = np.zeros(2)
    
    for l in range(2):
      if l == 0:
        nu = self.layers[l](torch.tensor(x)).detach().numpy().reshape(self.W[l].shape[0], 1)
      else:
        nu = self.layers[l](torch.tensor(omega.reshape(1, self.W[l].shape[1]))).detach().numpy().reshape(self.W[l].shape[0], 1)
      
      rht = self.rho[l] * self.eta[l]
      
      xtilde = (self.state.reshape(2,1) - self.xstar.reshape(2, 1))
      psitilde = nu - self.last_w[l]
      nutilde = nu - self.wstar[l]
      vec = np.vstack([xtilde, psitilde, nutilde])

      if xETM:
        mat = self.bigX[l]
      else:
        mat = np.block([
        [np.zeros((self.nx, self.nx)), np.zeros((self.nx, self.neurons[l])), np.zeros((self.nx, self.neurons[l]))],
        [self.Z[l], self.T[l], -self.T[l]],
        [np.zeros((self.neurons[l], self.nx)), np.zeros((self.neurons[l], self.neurons[l])), np.zeros((self.neurons[l], self.neurons[l]))]
      ])

      lht = (vec.T @ mat @ vec)[0][0]

      check = lht > rht

      if check:
        omega = func(torch.tensor(nu)).detach().numpy()
        self.last_w[l] = omega
        e[l] = 1
        psitilde = nu - omega
        vec = np.vstack([xtilde, psitilde, nutilde])
        lht = (vec.T @ mat @ vec)[0][0]
        val[l] = lht

      else:
        val[l] = lht
        omega = self.last_w[l]
    
    l = 2
    nu = self.layers[l](torch.tensor(omega.reshape(1, self.W[l].shape[1]))).detach().numpy()

    for i in range(2):
      self.eta[i] = (self.rho[i] + self.lam[i]) * self.eta[i] - val[i]
    
    return nu, e 
  
  def step(self):
    u, e = self.forward()
    x = (self.A + self.B @ self.K) @ self.state.reshape(2, 1) + self.B @ u.reshape(1, 1)   
    self.state = x
    return self.state, u, e, self.eta.tolist()

if __name__ == "__main__":
  import matplotlib.pyplot as plt

  s = Better_System()
  x0 = np.array([np.pi/6, 0.1])
  s.state = x0

  if xETM:
    P = np.load('Test/P.npy')
  else:
    P = np.load('mat-weights/P_try.npy')
  
  nsteps = 2500

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
    lyap.append((state - s.xstar.reshape(2, 1)).T @ P @ (state - s.xstar.reshape(2, 1)) + 2*eta[0] + 2*eta[1])

  states = np.array(states).squeeze()
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
      
  fig, axs = plt.subplots(3, 1)
  axs[0].plot(timegrid, inputs, label='Control input')
  axs[0].plot(timegrid, inputs * events[:, 1], marker='o', markerfacecolor='none', linestyle='None')
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


  fig, ax = ellipsoid_plot_2D(P, False, color='b', legend='ROA with dynamic ETM')
  ax.plot(states[:, 0], states[:, 1], 'b')
  ax.plot(s.xstar[0], s.xstar[1], marker='o', markersize=5, color='r')
  plt.legend()
  plt.show()