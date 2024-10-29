from systems_and_LMI.systems.NonLinearPendulum_NN_ETM import NonLinPendulum_NN_ETM
import systems_and_LMI.systems.nonlin_dynamic_ETM.params as params
import os
import numpy as np
import torch.nn as nn
import torch

class NonLinPendulum_NN_kETM(NonLinPendulum_NN_ETM):
  
  def __init__(self, reference=0.0):
    super().__init__(reference)

    Z_name = os.path.abspath(__file__ + '/../nonlin_dynamic_kETM/kZ.npy')
    T_name = os.path.abspath(__file__ + '/../nonlin_dynamic_kETM/kT.npy')
    Omega_name = os.path.abspath(__file__ + '/../nonlin_dynamic_kETM/Omega.npy')
    T = np.load(T_name)
    Z = np.load(Z_name)
    Omega = np.load(Omega_name)
    G = np.linalg.inv(T) @ Z
    self.G = np.split(G, self.nlayers - 1)
    Ck = np.zeros((self.nphi, self.nx))
    Ck[:, 0] = 1.0
    self.T = []
    self.Omega = []
    self.Ck = []
    for i in range(self.nlayers - 1):
      self.Omega.append(Omega[i*self.neurons[i]:(i+1)*self.neurons[i], i*self.neurons[i]:(i+1)*self.neurons[i]])
      self.Ck.append(Ck[i*self.neurons[i]:(i+1)*self.neurons[i], :])
      self.T.append(T[i*self.neurons[i]:(i+1)*self.neurons[i], i*self.neurons[i]:(i+1)*self.neurons[i]])
  
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
      vecr = (self.state - self.xstar).T @ self.Ck[l].T @ self.Omega[l] @ self.Ck[l] @ (self.state - self.xstar)

      check = (vec1 @ T @ vec2 - vecr > self.rho[l] * self.eta[l])[0][0]
      
      if check:
        omega = func(torch.tensor(nu)).detach().numpy()
        self.last_w[l] = nu
        e[l] = 1
        vec1 = (nu - omega).T
        vec2 = (self.G[l] @ (self.state - self.xstar) - (omega - self.wstar[l]))
        val[l] = (vec1 @ T @ vec2 - vecr)
      
      else:
        val[l] = (vec1 @ T @ vec2 - vecr)
        omega = self.last_w[l]
        
    l = self.nlayer - 1
    nu = self.layers[l](torch.tensor(omega.reshape(1, self.W[l].shape[1]))).detach().numpy().reshape(self.W[l].shape[0], 1)[0][0]

    for i in range(self.nlayer - 1):
      self.eta[i] = (self.rho[i] + self.lam[i]) * self.eta[i] - val[i]
    
    return nu, e
  
if __name__ == "__main__":
  
  s = NonLinPendulum_NN_ETM(0.3)
  x0 = np.array([[np.pi/3], [1.0], [0.0]])
  s.state = x0