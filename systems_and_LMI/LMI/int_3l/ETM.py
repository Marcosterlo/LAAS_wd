from systems_and_LMI.LMI.int_3l.main import LMI_3l_int
import systems_and_LMI.systems.nonlin_exp_ROA_kETM.params as params
import numpy as np
import cvxpy as cp
import warnings

class LMI_3l_int_ETM(LMI_3l_int):
  
  def __init__(self, W, b):
    super().__init__(W, b)

    gammavec = np.concatenate([params.gammas[i] * np.ones(self.neurons[i]) for i in range(self.nlayers)])
    gamma = cp.diag(gammavec)

    self.neurons = [32, 32, 32, 1]

    T_val = cp.Variable(self.nphi)
    self.T = cp.diag(T_val)
    self.T1 = self.T[:self.neurons[0], :self.neurons[0]]
    self.T2 = self.T[self.neurons[0]:self.neurons[0] + self.neurons[1], self.neurons[0]:self.neurons[0] + self.neurons[1]]
    self.T3 = self.T[self.neurons[0] + self.neurons[1]:self.neurons[0] + self.neurons[1] + self.neurons[2], self.neurons[0] + self.neurons[1]:self.neurons[0] + self.neurons[1] + self.neurons[2]]
    self.T4 = cp.reshape(self.T[-1, -1], (1, 1))

    self.Z = cp.Variable((self.nphi, self.nx))
    self.Z1 = self.Z[:self.neurons[0], :]
    self.Z2 = self.Z[self.neurons[0]:self.neurons[0] + self.neurons[1], :]
    self.Z3 = self.Z[self.neurons[0] + self.neurons[1]:self.neurons[0] + self.neurons[1] + self.neurons[2], :]
    self.Z4 = cp.reshape(self.Z[-1, :], (1, 3))

    self.Omega1 = cp.bmat([
      [np.zeros((self.nx, self.nx)), np.zeros((self.nx, self.neurons[0])), np.zeros((self.nx, self.neurons[0]))],
      [self.Z1, self.T1, -self.T1],
      [np.zeros((self.neurons[0], self.nx)), np.zeros((self.neurons[0], self.neurons[0])), np.zeros((self.neurons[0], self.neurons[0]))]
    ])
    
    self.Omega2 = cp.bmat([
      [np.zeros((self.nx, self.nx)), np.zeros((self.nx, self.neurons[1])), np.zeros((self.nx, self.neurons[1]))],
      [self.Z2, self.T2, -self.T2],
      [np.zeros((self.neurons[1], self.nx)), np.zeros((self.neurons[1], self.neurons[1])), np.zeros((self.neurons[1], self.neurons[1]))]
    ])
    
    self.Omega3 = cp.bmat([
      [np.zeros((self.nx, self.nx)), np.zeros((self.nx, self.neurons[2])), np.zeros((self.nx, self.neurons[2]))],
      [self.Z3, self.T3, -self.T3],
      [np.zeros((self.neurons[2], self.nx)), np.zeros((self.neurons[2], self.neurons[2])), np.zeros((self.neurons[2], self.neurons[2]))]
    ])

    self.Omega4 = cp.bmat([
      [np.zeros((self.nx, self.nx)), np.zeros((self.nx, self.neurons[3])), np.zeros((self.nx, self.neurons[3]))],
      [self.Z4, self.T4, -self.T4],
      [np.zeros((self.neurons[3], self.nx)), np.zeros((self.neurons[3], self.neurons[3])), np.zeros((self.neurons[3], self.neurons[3]))]
    ])

    # Constrain matrices definition
    self.Rphi = cp.bmat([
        [np.eye(self.nx), np.zeros((self.nx, self.nphi)), np.zeros((self.nx, self.nq))],
        [self.R @ self.Nvx, np.eye(self.R.shape[0]) - self.R, np.zeros((self.nphi, self.nq))],
        [np.zeros((self.nphi, self.nx)), np.eye(self.nphi), np.zeros((self.nphi, self.nq))],
        [np.zeros((self.nq, self.nx)), np.zeros((self.nq, self.nphi)), np.eye(self.nq)],
    ])
    
    self.M1 = cp.bmat([
      [np.zeros((self.nx, self.nx)), np.zeros((self.nx, self.nphi)), np.zeros((self.nx, self.nphi)), np.zeros((self.nx, self.nq))],
      [gamma @ self.Z, -gamma @ self.T , gamma @ self.T, np.zeros((self.nphi, self.nq))], 
      [np.zeros((self.nq, self.nx)), np.zeros((self.nq, self.nphi)), np.zeros((self.nq, self.nphi)), np.zeros((self.nq, self.nq))],
    ])
    
    self.Sinsec = cp.bmat([
      [0.0, -1.0],
      [-1.0, -2.0]
    ])
    
    self.Rs = cp.bmat([
      [np.array([[1.0, 0.0, 0.0]]), np.zeros((1, self.nphi)), np.zeros((1, self.nq))],
      [np.zeros((self.nq, self.nx)), np.zeros((1, self.nphi)), np.eye(self.nq)]
    ])

    self.M = cp.bmat([
      [self.Abar.T @ self.P @ self.Abar - self.P, self.Abar.T @ self.P @ self.Bbar, self.Abar.T @ self.P @ self.C],
      [self.Bbar.T @ self.P @ self.Abar, self.Bbar.T @ self.P @ self.Bbar, self.Bbar.T @ self.P @ self.C],
      [self.C.T @ self.P @ self.Abar, self.C.T @ self.P @ self.Bbar, self.C.T @ self.P @ self.C]
    ]) + self.M1 @ self.Rphi + self.Rphi.T @ self.M1.T + self.Rs.T @ self.Sinsec @ self.Rs

    # Constraints definiton
    self.constraints = [self.P >> 0]
    self.constraints += [self.T >> 0]
    self.constraints += [self.M << -self.m_thresh * np.eye(self.M.shape[0])]
    
    # Ellipsoid conditions for activation functions
    for i in range(self.nlayers - 1):
      for k in range(self.neurons[i]):
        Z_el = self.Z[i*self.neurons[i] + k]
        T_el = self.T[i*self.neurons[i] + k, i*self.neurons[i] + k]
        vcap = np.min([np.abs(-self.bound - self.wstar[i][k][0]), np.abs(self.bound - self.wstar[i][k][0])], axis=0)
        ellip = cp.bmat([
            [self.P, cp.reshape(Z_el, (self.nx ,1))],
            [cp.reshape(Z_el, (1, self.nx)), cp.reshape(2*self.alpha*T_el - self.alpha**2*vcap**(-2), (1, 1))] 
        ])
        self.constraints += [ellip >> 0]
    
    # Ellipsoid conditions for last saturation
    Z_el = self.Z[-1]
    T_el = self.T[-1, -1]
    vcap = np.min([np.abs(-self.bound - self.wstar[-1]), np.abs(self.bound - self.wstar[-1])], axis=0)
    ellip = cp.bmat([
        [self.P, cp.reshape(Z_el, (self.nx ,1))],
        [cp.reshape(Z_el, (1, self.nx)), cp.reshape(2*self.alpha*T_el - self.alpha**2*vcap**(-2), (1, 1))] 
    ])
    self.constraints += [ellip >> 0]
    
    # Objective function definition
    self.objective = cp.Minimize(cp.trace(self.P))

    # Problem definition
    self.prob = cp.Problem(self.objective, self.constraints)

    # User warnings filter
    warnings.filterwarnings("ignore", category=UserWarning, module='cvxpy')

if __name__ == "__main__":
  import os

  RL_weights = False
  
  if RL_weights:
    W1_name = os.path.abspath(__file__ + "/../../../systems/nonlin_norm_weights/mlp_extractor.policy_net.0.weight.csv")
    W2_name = os.path.abspath(__file__ + "/../../../systems/nonlin_norm_weights/mlp_extractor.policy_net.2.weight.csv")
    W3_name = os.path.abspath(__file__ + "/../../../systems/nonlin_norm_weights/mlp_extractor.policy_net.4.weight.csv")
    W4_name = os.path.abspath(__file__ + "/../../../systems/nonlin_norm_weights/action_net.weight.csv")
    
    b1_name = os.path.abspath(__file__ + "/../../../systems/nonlin_norm_weights/mlp_extractor.policy_net.0.bias.csv")
    b2_name = os.path.abspath(__file__ + "/../../../systems/nonlin_norm_weights/mlp_extractor.policy_net.2.bias.csv")
    b3_name = os.path.abspath(__file__ + "/../../../systems/nonlin_norm_weights/mlp_extractor.policy_net.4.bias.csv")
    b4_name = os.path.abspath(__file__ + "/../../../systems/nonlin_norm_weights/action_net.bias.csv")

  else:
    W1_name = os.path.abspath(__file__ + "/../../../../NN_training/int_nonlin/new_weights/mlp_extractor.policy_net.0.weight.csv")
    W2_name = os.path.abspath(__file__ + "/../../../../NN_training/int_nonlin/new_weights/mlp_extractor.policy_net.2.weight.csv")
    W3_name = os.path.abspath(__file__ + "/../../../../NN_training/int_nonlin/new_weights/mlp_extractor.policy_net.4.weight.csv")
    W4_name = os.path.abspath(__file__ + "/../../../../NN_training/int_nonlin/new_weights/action_net.weight.csv")

    b1_name = os.path.abspath(__file__ + "/../../../../NN_training/int_nonlin/new_weights/mlp_extractor.policy_net.0.bias.csv")
    b2_name = os.path.abspath(__file__ + "/../../../../NN_training/int_nonlin/new_weights/mlp_extractor.policy_net.2.bias.csv")
    b3_name = os.path.abspath(__file__ + "/../../../../NN_training/int_nonlin/new_weights/mlp_extractor.policy_net.4.bias.csv")
    b4_name = os.path.abspath(__file__ + "/../../../../NN_training/int_nonlin/new_weights/action_net.bias.csv")

    
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

  lmi = LMI_3l_int_ETM(W, b)
  # alpha = lmi.search_alpha(1, 0, 1e-5, verbose=True)
  alpha = 1
  P, T, Z = lmi.solve(alpha, verbose=True)
  # lmi.save_results('static_ETM')

  from systems_and_LMI.systems.NonLinPendulum_kETM_train_sat import NonLinPendulum_kETM_train_sat
  import matplotlib.pyplot as plt

  s = NonLinPendulum_kETM_train_sat(W, b, 0.0)
  Omega1 = lmi.Omega1.value 
  Omega2 = lmi.Omega2.value
  Omega3 = lmi.Omega3.value
  Omega4 = lmi.Omega4.value

  ref_bound = 5 * np.pi / 180
  in_ellip = False
  while not in_ellip:
    theta = np.random.uniform(-np.pi/2, np.pi/2)
    vtheta = np.random.uniform(-s.max_speed, s.max_speed)
    x0 = np.array([[theta], [vtheta], [0.0]])
    ref = np.random.uniform(-ref_bound, ref_bound)*0
    s = NonLinPendulum_kETM_train_sat(W, b, ref)
    if (x0).T @ P @ (x0) <= 1.0:
      in_ellip = True
      print(f"Initial state: theta0 = {theta*180/np.pi:.2f} deg, vtheta0 = {vtheta:.2f} rad/s, constant reference = {ref*180/np.pi:.2f} deg")
      s.state = x0
  
  s.mats = [Omega1, Omega2, Omega3, Omega4]
  
  nsteps = 500

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
    lyap.append((state - s.xstar).T @ P @ (state - s.xstar) + 2*eta[0] + 2*eta[1] + 2*eta[2] + 2*eta[3])

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
      
  fig, axs = plt.subplots(4, 1)
  axs[0].plot(timegrid, inputs, label='Control input')
  axs[0].plot(timegrid, inputs * events[:, 3], marker='o', markerfacecolor='none', linestyle='None')
  axs[0].plot(timegrid, timegrid * 0 + s.wstar[-1] * s.max_torque, 'r--')
  axs[0].set_xlabel('Time steps')
  axs[0].set_ylabel('Values')
  axs[0].legend()
  axs[0].grid(True)

  axs[1].plot(timegrid, states[:, 0], label='Position')
  axs[1].plot(timegrid, states[:, 0] * events[:, 3], marker='o', markerfacecolor='none', linestyle='None')
  axs[1].plot(timegrid, timegrid * 0 + s.xstar[0], 'r--')
  axs[1].set_xlabel('Time steps')
  axs[1].set_ylabel('Values')
  axs[1].legend()
  axs[1].grid(True)

  axs[2].plot(timegrid, states[:, 1], label='Velocity')
  axs[2].plot(timegrid, states[:, 1] * events[:, 3], marker='o', markerfacecolor='none', linestyle='None')
  axs[2].plot(timegrid, timegrid * 0 + s.xstar[1], 'r--')
  axs[2].set_xlabel('Time steps')
  axs[2].set_ylabel('Values')
  axs[2].legend()
  axs[2].grid(True)

  axs[3].plot(timegrid, states[:, 2], label='Integrator state')
  axs[3].plot(timegrid, states[:, 2] * events[:, 3], marker='o', markerfacecolor='none', linestyle='None')
  axs[3].plot(timegrid, timegrid * 0 + s.xstar[2], 'r--')
  axs[3].set_xlabel('Time steps')
  axs[3].set_ylabel('Values')
  axs[3].legend()
  axs[3].grid(True)
  plt.show()

  plt.plot(timegrid, etas[:, 0], label='Eta_1')
  plt.plot(timegrid, etas[:, 1], label='Eta_2')
  plt.plot(timegrid, etas[:, 2], label='Eta_3')
  plt.plot(timegrid, etas[:, 3], label='Eta_4')
  plt.legend()
  plt.grid(True)
  plt.show()

  plt.plot(timegrid, lyap, label='Lyapunov function')
  plt.legend()
  plt.grid(True)
  plt.show()