from LinearPendulum_integrator import LinPendulumIntegrator
import numpy as np
import os
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

# New class definition that includes dynamic ETM
class LinPendulumIntegrator_ETM(LinPendulumIntegrator):
  
  # Calling the constructor of the parent class
  def __init__(self):
    super().__init__()
    # Adding the new parameters regarding the dyanic ETM
    T_mat_name = os.path.abspath(__file__ + "/../dynamic_ETM/T.npy")
    Z_mat_name = os.path.abspath(__file__ + "/../dynamic_ETM/Z.npy")
    P_mat_name = os.path.abspath(__file__ + "/../dynamic_ETM/P.npy")
    T = np.load(T_mat_name)
    Z = np.load(Z_mat_name)
    self.P = np.load(P_mat_name)
    G = np.linalg.inv(T) @ Z
    self.G = np.split(G, self.nlayer - 1)
    self.T = []
    for i in range(self.nlayer - 1):
      self.T.append(T[i*self.neurons[i]:(i+1)*self.neurons[i], i*self.neurons[i]:(i+1)*self.neurons[i]])
    self.eta = np.ones(self.nlayer - 1)
    self.rho = np.array([0.4, 0.3, 0.2])
    self.lam = np.array([0.5, 0.6, 0.7])
    self.last_w = []
    for i, neuron in enumerate(self.neurons):
      self.last_w.append(np.ones((neuron, 1))*1e3)
  
  # Override the forward method as now it includes the ETM mechanism
  def forward(self, dynamic=True, no_ETM=False):

    # Definition of activation function
    func = nn.Hardtanh()

    # Variable to store sector values
    val = np.zeros(self.nlayer - 1)

    # Empty vector to store events
    e = np.zeros(self.nlayer - 1)

    # Reshape of state according to NN dimensions
    x = self.state.reshape(1, self.W[0].shape[1])

    # Loop over the layers
    for l in range(self.nlayer - 1):

      # If we are in the first layer the input is the state, otherwise it is the omega of the previous layer
      if l == 0:
        nu = self.layers[l](torch.tensor(x)).detach().numpy().reshape(self.W[l].shape[0], 1)
      else:
        nu = self.layers[l](torch.tensor(omega.reshape(1, self.W[l].shape[1]))).detach().numpy().reshape(self.W[l].shape[0], 1)

      # ETM evaluation
      vec1 = (nu - self.last_w[l]).T
      T = self.T[l]
      vec2 = (self.G[l] @ (self.state - self.xstar) - (self.last_w[l] - self.wstar[l]))

      if dynamic:
        check = vec1 @ T @ vec2 > self.rho[l] * self.eta[l] 
      elif no_ETM:
        check = True
      else:
        check = vec1 @ T @ vec2 > 0
        

      if check:
        # If there is an event we compute the output of the layer with the non linear activation function
        omega = func(torch.tensor(nu)).detach().numpy()
        # We substitute the last computed value
        self.last_w[l] = omega
        # I fals the event at layer l
        e[l] = 1
        # The new sector conditions are now evaluated
        vec1 = (nu - omega).T
        vec2 = (self.G[l] @ (x - self.xstar) - (omega - self.wstar[l]))
        val[l] = (vec1 @ T @ vec2)[0][0]
      
      else:
        val[l] = (vec1 @ T @ vec2)[0][0]
        # If no event occurs we feed the next layer the last stored output
        omega = self.last_w[l]
    
    # Last layer, it doesn't have the ETM mechanism and the activation function
    l = self.nlayer - 1
    nu = self.layers[l](torch.tensor(omega.reshape(1, self.W[l].shape[1]))).detach().numpy().reshape(self.W[l].shape[0], 1)

    # Eta dynamics
    for i in range(self.nlayer - 1):
      self.eta[i] = (self.rho[i] + self.lam[i]) * self.eta[i] - val[i]
    
    return nu, e, self.eta

  # Override the step method as now it includes the ETM mechanism
  def step(self, dynamic=True, no_ETM=False):

    # Compute the output of the network
    u, e, eta = self.forward(dynamic=dynamic, no_ETM=no_ETM)

    # Compute the new state
    newx = self.A @ self.state.reshape(self.nx, 1) + self.B @ u
    newx[2] += -self.constant_reference
    self.state = newx

    return self.state, u, e, eta

if __name__ == "__main__":
  s = LinPendulumIntegrator_ETM()
  x0 = np.array([[0.2], [1.0], [0.0]])
  s.state = x0
  
  states = []
  inputs = []
  events = []
  etas = []

  nsteps = 1000
  s.constant_reference = 0.1

  for i in range(nsteps):
    # state, u, e, eta = s.step()
    # state, u, e, eta = s.step(dynamic=False)
    state, u, e, eta = s.step(dynamic=False, no_ETM=True)
    states.append(state)
    inputs.append(u)
    events.append(e)
    etas.append(eta)
  
  states = np.array(states)
  inputs = np.array(inputs)
  etas = np.array(etas)
  events = np.array(events)

  layer1_trigger = np.sum(events[:, 0]) / nsteps * 100
  layer2_trigger = np.sum(events[:, 1]) / nsteps * 100
  layer3_trigger = np.sum(events[:, 2]) / nsteps * 100

  print(f"Layer 1 has been triggered {layer1_trigger:.1f}% of times")
  print(f"Layer 2 has been triggered {layer2_trigger:.1f}% of times")
  print(f"Layer 3 has been triggered {layer3_trigger:.1f}% of times")

  for i, event in enumerate(events):
    if not event[0]:
      events[i][0] = None
    if not event[1]:
      events[i][1] = None
    if not event[2]:
      events[i][2] = None
  events = np.squeeze(events)

  timegrid = np.arange(0, nsteps)

  plt.plot(timegrid, states[:, 0])
  plt.grid(True)
  plt.show()

  plt.plot(timegrid, states[:, 1])
  plt.grid(True)
  plt.show()

  plt.plot(timegrid, states[:, 2])
  plt.grid(True)
  plt.show()

  inputs = np.squeeze(inputs)
  plt.plot(timegrid, inputs)
  plt.plot(timegrid, np.squeeze((events[:, 2]*inputs).reshape(len(timegrid))), marker='o', markerfacecolor='none')
  plt.grid(True)
  plt.show()