from systems_and_LMI.systems.NonLinPendulum_train import NonLinPendulum_train
import systems_and_LMI.systems.nonlin_exp_ROA_kETM_sat.params as params
import numpy as np
import os
import torch.nn as nn
import torch
import warnings

# User warnings filter
warnings.filterwarnings("ignore", category=UserWarning, module='torch')

# New class definition that depends on the passed weights and biases and implements ETMs
class NonLinPendulum_kETM_train_sat(NonLinPendulum_train):
  
  def __init__(self, W, b, ref):
    super().__init__(W, b, ref)

    Z_name = os.path.abspath(__file__ + '/../nonlin_exp_ROA_kETM_sat/Z.npy')
    T_name = os.path.abspath(__file__ + '/../nonlin_exp_ROA_kETM_sat/T.npy')

    T = np.load(T_name)
    Z = np.load(Z_name)
    self.Z = np.split(Z, [32, 64, 96])
    self.T = []
    self.neurons = [32, 32, 32, 1]
    for i in range(self.nlayers):
      self.T.append(T[i*self.neurons[i]:(i+1)*self.neurons[i], i*self.neurons[i]:(i+1)*self.neurons[i]])
    self.eta = np.ones(self.nlayers)*params.eta0
    self.rho = params.rhos
    self.lam = params.lambdas
    self.last_w = []
    for i, neuron in enumerate(self.neurons):
      self.last_w.append(np.ones((neuron, 1))*1e3)

  def forward(self):
    func = nn.Hardtanh()
    e = np.zeros(self.nlayers)
    x = self.state.reshape(1, self.W[0].shape[1]) 
    val = np.zeros(self.nlayers)

    for l in range(self.nlayers):
      if l == 0:
        nu = self.layers[l](torch.tensor(x)).detach().numpy().reshape(self.W[l].shape[0], 1)
      else:
        nu = self.layers[l](torch.tensor(omega.reshape(1, self.W[l].shape[1]))).detach().numpy().reshape(self.W[l].shape[0], 1)

      rht = self.rho[l] * self.eta[l]
      
      xtilde = self.state - self.xstar
      psitilde = nu - self.last_w[l]
      nutilde = nu - self.wstar[l]
      vec = np.vstack([xtilde, psitilde, nutilde])
      sec_mat = np.block([
        [np.zeros((self.nx, self.nx)), np.zeros((self.nx, self.neurons[l])), np.zeros((self.nx, self.neurons[l]))],
        [self.Z[l], self.T[l], -self.T[l]],
        [np.zeros((self.neurons[l], self.nx)), np.zeros((self.neurons[l], self.neurons[l])), np.zeros((self.neurons[l], self.neurons[l]))]
      ])

      lht = (vec.T @ sec_mat @ vec)[0][0]

      check = lht  > rht
      
      if check:
        omega = func(torch.tensor(nu)).detach().numpy()
        self.last_w[l] = omega
        e[l] = 1
        psitilde = nu - omega
        vec = np.vstack([xtilde, psitilde, nutilde])
        lht = (vec.T @ sec_mat @ vec)[0][0]
        val[l] = lht
      
      else:
        val[l] = lht
        omega = self.last_w[l]
        
    for i in range(self.nlayer - 1):
      self.eta[i] = (self.rho[i] + self.lam[i]) * self.eta[i] - val[i]
      
    return omega, e
  
  def step(self):
    u, e = self.forward()
    u = np.clip(u, -1.0, 1.0)*self.max_torque
    nonlin = np.sin(self.state[0]) - self.state[0]
    self.state = self.A @ self.state + self.B @ u + self.C * nonlin + self.D * self.constant_reference
    etaval = self.eta.tolist()
    return self.state, u, e, etaval

if __name__ == "__main__":

  W1_name = os.path.abspath(__file__ + "/../nonlin_exp_ROA_kETM/mlp_extractor.policy_net.0.weight.csv")
  W2_name = os.path.abspath(__file__ + "/../nonlin_exp_ROA_kETM/mlp_extractor.policy_net.2.weight.csv")
  W3_name = os.path.abspath(__file__ + "/../nonlin_exp_ROA_kETM/mlp_extractor.policy_net.4.weight.csv")
  W4_name = os.path.abspath(__file__ + "/../nonlin_exp_ROA_kETM/action_net.weight.csv")
  
  b1_name = os.path.abspath(__file__ + "/../nonlin_exp_ROA_kETM/mlp_extractor.policy_net.0.bias.csv")
  b2_name = os.path.abspath(__file__ + "/../nonlin_exp_ROA_kETM/mlp_extractor.policy_net.2.bias.csv")
  b3_name = os.path.abspath(__file__ + "/../nonlin_exp_ROA_kETM/mlp_extractor.policy_net.4.bias.csv")
  b4_name = os.path.abspath(__file__ + "/../nonlin_exp_ROA_kETM/action_net.bias.csv")

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

  ref = 0.3

  s = NonLinPendulum_kETM_train_sat(W, b, ref)
  s.state = np.array([[np.pi/4], [3.0], [0.0]])

  states = []
  inputs = []
  events = []
  etas = []
  
  nsteps = 300
  for i in range(nsteps):
    state, u, e, eta = s.step()
    states.append(state)
    inputs.append(u)
    events.append(e)
    etas.append(eta)
  
  states = np.squeeze(np.array(states))
  inputs = np.squeeze(np.array(inputs))
  events = np.squeeze(np.array(events))
  etas = np.squeeze(np.array(etas))
  timegrid = np.arange(0, nsteps)

  layer1_trigger = np.sum(events[:, 0]) / nsteps * 100
  layer2_trigger = np.sum(events[:, 1]) / nsteps * 100
  layer3_trigger = np.sum(events[:, 2]) / nsteps * 100
  layer4_trigger = np.sum(events[:, 3]) / nsteps * 100
  
  print(f"Layer 1 has been triggered {layer1_trigger:.1f}% of times")
  print(f"Layer 2 has been triggered {layer2_trigger:.1f}% of times")
  print(f"Layer 3 has been triggered {layer3_trigger:.1f}% of times")
  print(f"Output layer has been triggered {layer4_trigger:.1f}% of times")
  
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
   
  plt.plot(timegrid, states[:, 0])
  plt.plot(timegrid, states[:, 0]*events[:,3], 'o', markerfacecolor='none', linestyle='None')
  plt.plot(timegrid, timegrid*0.0 + s.xstar[0], 'r--')
  plt.grid(True)
  plt.show()

  plt.plot(timegrid, states[:, 1])
  plt.plot(timegrid, states[:, 1]*events[:, 3], marker='o', markerfacecolor='none', linestyle='None')
  plt.plot(timegrid, timegrid*0.0 + s.xstar[1], 'r--')
  plt.grid(True)
  plt.show()

  plt.plot(timegrid, inputs)
  plt.plot(timegrid, inputs*events[:, 3], marker='o', markerfacecolor='none', linestyle='None')
  plt.grid(True)
  plt.show()