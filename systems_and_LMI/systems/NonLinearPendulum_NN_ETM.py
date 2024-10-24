from systems_and_LMI.systems.NonLinearPendulum_NN import NonLinPendulum_NN
import systems_and_LMI.systems.nonlin_dynamic_ETM.params as params
import os
import numpy as np
import torch.nn as nn
import torch

class NonLinPendulum_NN_ETM(NonLinPendulum_NN):
  
  def __init__(self):
    super().__init__()

    Z_name = os.path.abspath(__file__ + '/../nonlin_dynamic_ETM/Z.npy')
    T_name = os.path.abspath(__file__ + '/../nonlin_dynamic_ETM/T.npy')
    T = np.load(T_name)
    Z = np.load(Z_name)
    G = np.linalg.inv(T) @ Z
    self.G = np.split(G, self.nlayers - 1)
    self.T = []
    for i in range(self.nlayers - 1):
      self.T.append(T[i*self.neurons[i]:(i+1)*self.neurons[i], i*self.neurons[i]:(i+1)*self.neurons[i]])
    self.eta = np.ones(self.nlayers - 1)*0
    self.rho = params.rhos
    self.lam = params.lambdas
    self.last_w = []
    for i, neuron in enumerate(self.neurons):
      self.last_w.append(np.ones((neuron, 1))*1e3)
  
  def forward(self):
    func = nn.Hardtanh()
    e = np.zeros(self.nlayer - 1)
    x = self.state.reshape(1, self.W[0].shape[1]) 
    val = np.zeros(self.nlayer - 1)

    for l in range(self.nlayer - 1):
      if l == 0:
        nu = self.layers[l](torch.tensor(x)).detach().numpy().reshape(self.W[l].shape[0], 1)
      else:
        nu = self.layers[l](torch.tensor(omega.reshape(1, self.W[l].shape[1]))).detach().numpy().reshape(self.W[l].shape[0], 1)
      
      vec1 = (nu - self.last_w[l]).T
      T = self.T[l]
      vec2 = (self.G[l] @ (self.state - self.xstar) - (self.last_w[l] - self.wstar[l]))

      check = (vec1 @ T @ vec2 > self.rho[l] * self.eta[l])[0][0]
      
      if check:
        omega = func(torch.tensor(nu)).detach().numpy()
        self.last_w[l] = nu
        e[l] = 1
        vec1 = (nu - omega).T
        vec2 = (self.G[l] @ (self.state - self.xstar) - (omega - self.wstar[l]))
        val[l] = (vec1 @ T @ vec2)
      
      else:
        val[l] = (vec1 @ T @ vec2)
        omega = self.last_w[l]
        
    l = self.nlayer - 1
    nu = self.layers[l](torch.tensor(omega.reshape(1, self.W[l].shape[1]))).detach().numpy().reshape(self.W[l].shape[0], 1)[0][0]

    for i in range(self.nlayer - 1):
      self.eta[i] = (self.rho[i] + self.lam[i]) * self.eta[i] - val[i]
    
    return nu, e
  
  def step(self):
    u, e = self.forward()
    nonlin = np.sin(self.state[0]) - self.state[0]
    self.state = self.A @ self.state + self.B * u + self.C * nonlin
    self.state[2] += -self.constant_reference
    # Copy of eta since it apparently returns a pointer to the class parameter
    etaval = self.eta.tolist()
    return self.state, u, e, etaval

if __name__ == "__main__":
  
  s = NonLinPendulum_NN_ETM()
  x0 = np.array([[np.pi/3], [1.0], [0.0]])
  s.state = x0