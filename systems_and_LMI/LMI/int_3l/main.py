from systems_and_LMI.systems.NonLinPendulum_train import NonLinPendulum_train
import params
import numpy as np
import cvxpy as cp
import os
import warnings

class LMI_3l_int():
  def __init__(self, W, b):
    
    self.system = NonLinPendulum_train(W, b, 0.0)
    self.A = self.system.A
    self.max_torque = self.system.max_torque
    self.B = self.system.B
    self.C = self.system.C
    self.nx = self.system.nx
    self.nq = self.system.nq
    self.bound = 1
    self.xstar = self.system.xstar
    self.wstar = self.system.wstar
    self.R = self.system.R
    self.Rw = self.system.Rw
    self.Rb = self.system.Rb
    self.Nux = self.system.N[0]
    self.Nuw = self.system.N[1]
    self.Nub = self.system.N[2]
    self.Nvx = self.system.N[3]
    self.Nvw = self.system.N[4]
    self.Nvb = self.system.N[5]
    self.nphi = self.system.nphi
    self.nlayers = self.system.nlayers
    self.neurons = [32, 32, 32, 1]
    self.nu = 1
    self.gammas = params.gammas
    self.gamma1_scal = self.gammas[0]
    self.gamma2_scal = self.gammas[1]
    self.gamma3_scal = self.gammas[2]
    self.gamma4_scal = -1.0
    self.nbigx = self.nx + self.neurons[0] * 2
    self.dynamic = True
    self.xETM = False
    
    # Constraint related parameters
    self.m_thresh = 1e-6
    
    # Auxiliary parameters
    self.Abar = self.A + self.B @ self.Rw
    self.Bbar = -self.B @ self.Nuw @ self.R
    
    # Variables definition
    self.P = cp.Variable((self.nx, self.nx), symmetric=True)
    T_val = cp.Variable(self.nphi)
    self.T = cp.diag(T_val)
    self.T1 = self.T[:self.neurons[0], :self.neurons[0]]
    self.T2 = self.T[self.neurons[0]:self.neurons[0] + self.neurons[1], self.neurons[0]:self.neurons[0] + self.neurons[1]]
    self.T3 = self.T[self.neurons[0] + self.neurons[1]:self.neurons[0] + self.neurons[1] + self.neurons[2], self.neurons[0] + self.neurons[1]:self.neurons[0] + self.neurons[1] + self.neurons[2]]
    self.T_sat = cp.reshape(self.T[-1, -1], (self.nu, self.nu))
    
    self.Z = cp.Variable((self.nphi, self.nx))
    self.Z1 = self.Z[:self.neurons[0], :]
    self.Z2 = self.Z[self.neurons[0]:self.neurons[0] + self.neurons[1], :]
    self.Z3 = self.Z[self.neurons[0] + self.neurons[1]:self.neurons[0] + self.neurons[1] + self.neurons[2], :]
    self.Z_sat = cp.reshape(self.Z[-1, :], (self.nu, self.nx))
    
    # New ETM matrices
    self.bigX1 = cp.Variable((self.nbigx, self.nbigx))
    self.bigX2 = cp.Variable((self.nbigx, self.nbigx))
    self.bigX3 = cp.Variable((self.nbigx, self.nbigx))
    
    # Finsler multipliers
    self.N11 = cp.Variable((self.nx, self.nphi))
    self.N12 = cp.Variable((self.nphi, self.nphi), symmetric=True)
    N13 = cp.Variable(self.nphi)
    self.N13 = cp.diag(N13)
    self.N1 = cp.vstack([self.N11, self.N12, self.N13])

    self.N21 = cp.Variable((self.nx, self.nphi))
    self.N22 = cp.Variable((self.nphi, self.nphi), symmetric=True)
    N23 = cp.Variable(self.nphi)
    self.N23 = cp.diag(N23)
    self.N2 = cp.vstack([self.N21, self.N22, self.N23])
    
    self.N31 = cp.Variable((self.nx, self.nphi))
    self.N32 = cp.Variable((self.nphi, self.nphi), symmetric=True)
    N33 = cp.Variable(self.nphi)
    self.N33 = cp.diag(N33)
    self.N3 = cp.vstack([self.N31, self.N32, self.N33])
    
    # Parameters definition
    self.alpha = cp.Parameter(nonneg=True)
    
    # Constrain matrices definition
    self.Rphi = cp.bmat([
        [np.eye(self.nx), np.zeros((self.nx, self.nphi)), np.zeros((self.nx, self.nq))],
        [self.R @ self.Nvx, np.eye(self.R.shape[0]) - self.R, np.zeros((self.nphi, self.nq))],
        [np.zeros((self.nphi, self.nx)), np.eye(self.nphi), np.zeros((self.nphi, self.nq))],
        [np.zeros((self.nq, self.nx)), np.zeros((self.nq, self.nphi)), np.eye(self.nq)],
    ])
    
    self.M1 = cp.bmat([
      [np.zeros((self.nx, self.nx)), np.zeros((self.nx, self.nphi)), np.zeros((self.nx, self.nphi)), np.zeros((self.nx, self.nq))],
      [self.Z, -self.T , self.T, np.zeros((self.nphi, self.nq))], 
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
    
    idx = np.eye(self.nx)
    xzero = np.zeros((self.nx, self.neurons[0]))
    xzeros = np.zeros((self.nx, self.nu))

    id = np.eye(self.neurons[0])
    zero = np.zeros((self.neurons[0], self.neurons[0]))
    zerox = np.zeros((self.neurons[0], self.nx))
    zeros = np.zeros((self.neurons[0], self.nu))

    ids = np.eye(self.nu)
    szerox = np.zeros((self.nu, self.nx))
    szero = np.zeros((self.nu, self.neurons[0]))
    szeros = np.zeros((self.nu, self.nu))

    self.R1 = cp.bmat([
      [idx, xzero, xzero, xzero, xzeros, xzero, xzero, xzero, xzeros],
      [zerox, id, zero, zero, zeros, zero, zero, zero, zeros],
      [zerox, zero, zero, zero, zeros, id, zero, zero, zeros],
    ])

    self.R2 = cp.bmat([
      [idx, xzero, xzero, xzero, xzeros, xzero, xzero, xzero, xzeros],
      [zerox, zero, id, zero, zeros, zero, zero, zero, zeros],
      [zerox, zero, zero, zero, zeros, zero, id, zero, zeros],
    ])

    self.R3 = cp.bmat([
      [idx, xzero, xzero, xzero, xzeros, xzero, xzero, xzero, xzeros],
      [zerox, zero, zero, id, zeros, zero, zero, zero, zeros],
      [zerox, zero, zero, zero, zeros, zero, zero, id, zeros],
    ])

    self.Rsat = cp.bmat([
      [idx, xzero, xzero, xzero, xzeros, xzero, xzero, xzero, xzeros],
      [szerox, szero, szero, szero, ids, szero, szero, szero, szeros],
      [szerox, szero, szero, szero, szeros, szero, szero, szero, ids]
    ])

    # self.Rsat = cp.bmat([
    #   [np.eye(self.nx), np.zeros((self.nx, self.nphi)), np.zeros((self.nx, self.nq))],
    #   [np.zeros((self.nu, self.nx)), np.zeros((self.nu, self.nphi)), np.zeros((self.nu, self.nq))],
    #   [self.Rw, -self.Nuw @ self.R, np.zeros((self.nu, self.nq))],
    # ])

    # Transformation matrix to pass from xi = [x, psi1, psi2, psi3, nu1, nu2, nu3] to [x, psi1, psi2, psi3]
    self.Rnu = cp.bmat([
      [np.eye(self.nx), np.zeros((self.nx, self.nphi)), np.zeros((self.nx, self.nq))],
      [np.zeros((self.nphi, self.nx)), np.eye(self.nphi), np.zeros((self.nphi, self.nq))],
      [self.R @ self.Nvx, np.eye(self.R.shape[0]) - self.R, np.zeros((self.nphi, self.nq))],
    ])

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

    self.Omegas = cp.bmat([
      [np.zeros((self.nx, self.nx)), np.zeros((self.nx, self.nu)), np.zeros((self.nx, self.nu))],
      [self.Z_sat, self.T_sat, -self.T_sat],
      [np.zeros((self.nu, self.nx)), np.zeros((self.nu, self.nu)), np.zeros((self.nu, self.nu))]
    ])
    
    if self.dynamic:
      if self.xETM:
        self.M2 = self.Rnu.T @ (self.R1.T @ (self.gamma1_scal * (self.bigX1 + self.bigX1.T)) @ self.R1 + self.R2.T @ (self.gamma2_scal * (self.bigX2 + self.bigX2.T)) @ self.R2 + self.R3.T @ (self.gamma3_scal * (self.bigX3 + self.bigX3.T)) @ self.R3 + self.Rsat.T @ (self.gamma4_scal * (self.Omegas + self.Omegas.T)) @ self.Rsat) @ self.Rnu
      else:
        self.M2 = self.Rnu.T @ (self.R1.T @ (self.gamma1_scal * (self.Omega1 + self.Omega1.T)) @ self.R1 + self.R2.T @ (self.gamma2_scal * (self.Omega2 + self.Omega2.T)) @ self.R2 + self.R3.T @ (self.gamma3_scal * (self.Omega3 + self.Omega3.T)) @ self.R3 + self.Rsat.T @ (self.gamma4_scal *(self.Omegas + self.Omegas.T)) @ self.Rsat) @ self.Rnu
    else: 
      self.M2 = -self.Rnu.T @ (self.R1.T @ (self.Omega1 + self.Omega1.T) @ self.R1 + self.R2.T @ (self.Omega2 + self.Omega2.T) @ self.R2 + self.R3.T @ (self.Omega3 + self.Omega3.T) @ self.R3 + self.Rsat.T @ (self.Omegas + self.Omegas.T) @ self.Rsat) @ self.Rnu
    

    self.M = cp.bmat([
      [self.Abar.T @ self.P @ self.Abar - self.P, self.Abar.T @ self.P @ self.Bbar, self.Abar.T @ self.P @ self.C],
      [self.Bbar.T @ self.P @ self.Abar, self.Bbar.T @ self.P @ self.Bbar, self.Bbar.T @ self.P @ self.C],
      [self.C.T @ self.P @ self.Abar, self.C.T @ self.P @ self.Bbar, self.C.T @ self.P @ self.C]
    ]) + self.M2 + self.Rs.T @ self.Sinsec @ self.Rs
    
    # Constraint matrices definition
    # Finsler constraint to handle nu with respect to x and psi
    self.hconstr = cp.hstack([self.R @ self.Nvx, np.eye(self.R.shape[0]) - self.R, -np.eye(self.nphi)])

    # Finsler constraints
    self.finsler1 = self.R1.T @ (self.bigX1 - self.Omega1 + self.bigX1.T - self.Omega1.T) @ self.R1 + self.N1 @ self.hconstr + self.hconstr.T @ self.N1.T

    self.finsler2 = self.R2.T @ (self.bigX2 - self.Omega2 + self.bigX2.T - self.Omega2.T) @ self.R2 + self.N2 @ self.hconstr + self.hconstr.T @ self.N2.T
    
    self.finsler3 = self.R3.T @ (self.bigX3 - self.Omega3 + self.bigX3.T - self.Omega3.T) @ self.R3 + self.N3 @ self.hconstr + self.hconstr.T @ self.N3.T

    # Constraints definiton
    self.constraints = [self.P >> 0]
    self.constraints += [self.T >> 0]
    self.constraints += [self.M << -self.m_thresh * np.eye(self.M.shape[0])]
    if self.xETM:
      self.constraints += [self.finsler1 << 0]
      self.constraints += [self.finsler2 << 0]
      self.constraints += [self.finsler3 << 0]
    
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
    Z_el = self.Z_sat
    T_el = self.T_sat
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

  def solve(self, alpha_val, verbose=False):
    self.alpha.value = alpha_val
    try:
      self.prob.solve(solver=cp.MOSEK, verbose=True)
    except cp.error.SolverError:
      return None, None, None

    if self.prob.status not in ["optimal", "optimal_inaccurate"]:
      return None, None, None
    else:
      if verbose:
        print(f"Max eigenvalue of P: {np.max(np.linalg.eigvals(self.P.value))}")
        print(f"Max eigenvalue of M: {np.max(np.linalg.eigvals(self.M.value))}") 
      return self.P.value, self.T.value, self.Z.value
  
  def search_alpha(self, feasible_extreme, infeasible_extreme, threshold, verbose=False):
    golden_ratio = (1 + np.sqrt(5)) / 2
    i = 0
    while (feasible_extreme - infeasible_extreme > threshold) and i < 11:
      i += 1
      alpha1 = feasible_extreme - (feasible_extreme - infeasible_extreme) / golden_ratio
      alpha2 = infeasible_extreme + (feasible_extreme - infeasible_extreme) / golden_ratio
      
      P1, _, _ = self.solve(alpha1, verbose=False)
      if P1 is None:
        val1 = 1e10
      else:
        val1 = np.max(np.linalg.eigvals(P1))
      
      P2, _, _ = self.solve(alpha2, verbose=False)
      if P2 is None:
        val2 = 1e10
      else:
        val2 = np.max(np.linalg.eigvals(P2))
        
      if val1 < val2:
        feasible_extreme = alpha2
      else:
        infeasible_extreme = alpha1
        
      if verbose:
        if val1 < val2:
          P_eig = val1
        else:
          P_eig = val2
        print(f"\nIteration number: {i}")
        print(f"==================== \nMax eigenvalue of P: {P_eig}")
        print(f"Current alpha value: {feasible_extreme}\n==================== \n")
    
    return feasible_extreme
  
  def save_results(self, path_dir: str):
    if not os.path.exists(path_dir):
      os.makedirs(path_dir)
    P, T, Z = self.solve(self.alpha.value)
    np.save(f"{path_dir}/P.npy", P)
    np.save(f"{path_dir}/T.npy", T)
    np.save(f"{path_dir}/Z.npy", Z)
    return P, T, Z

if __name__ == "__main__":

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

  lmi = LMI_3l_int(W, b)
  # alpha = lmi.search_alpha(1, 0, 1e-5, verbose=True)
  if lmi.dynamic:
    alpha = 0.05
  else:
    alpha = 0.0558
  P, T, Z = lmi.solve(alpha, verbose=True)
  # lmi.save_results('static_ETM')
  
  if P is not None: 
    print(f"Size of ROA: {np.pi/np.sqrt(np.linalg.det(P)):.2f}")

  from systems_and_LMI.systems.FinalPendulum_xETM import FinalPendulum_xETM
  import matplotlib.pyplot as plt

  s = FinalPendulum_xETM(W, b, 0.0)
  if lmi.xETM:
    Omega1 = lmi.bigX1.value
    Omega2 = lmi.bigX2.value
    Omega3 = lmi.bigX3.value
  else:
    Omega1 = lmi.Omega1.value 
    Omega2 = lmi.Omega2.value
    Omega3 = lmi.Omega3.value

  ref_bound = 5 * np.pi / 180
  in_ellip = False
  while not in_ellip:
    theta = np.random.uniform(-np.pi/2, np.pi/2)
    vtheta = np.random.uniform(-s.max_speed, s.max_speed)
    x0 = np.array([[theta], [vtheta], [0.0]])
    ref = np.random.uniform(-ref_bound, ref_bound)
    s = FinalPendulum_xETM(W, b, ref)
    if (x0).T @ P @ (x0) <= 1.0:
      in_ellip = True
      print(f"Initial state: theta0 = {theta*180/np.pi:.2f} deg, vtheta0 = {vtheta:.2f} rad/s, constant reference = {ref*180/np.pi:.2f} deg")
      s.state = x0
  
  if not lmi.dynamic:
    s.rho *= 0.0 
  
  s.bigX = [Omega1, Omega2, Omega3]
  
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
    lyap.append((state - s.xstar).T @ P @ (state - s.xstar) + 2*eta[0] + 2*eta[1] + 2*eta[2])

  states = np.insert(states, 0, x0, axis=0)
  states = np.delete(states, -1, axis=0)
  states = np.squeeze(np.array(states))
  states[:, 0] *= 180 / np.pi
  s.xstar[0] *= 180 / np.pi

  inputs = np.insert(inputs, 0, np.array(0.0), axis=0)
  inputs = np.delete(inputs, -1, axis=0)
  inputs = np.squeeze(np.array(inputs))*s.max_torque

  events = np.squeeze(np.array(events))
  etas = np.squeeze(np.array(etas))
  lyap = np.squeeze(np.array(lyap))

  timegrid = np.arange(0, nsteps)

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
      
  fig, axs = plt.subplots(4, 1)
  axs[0].plot(timegrid, inputs, label='Control input')
  axs[0].plot(timegrid, inputs * events[:, 2], marker='o', markerfacecolor='none', linestyle='None')
  axs[0].plot(timegrid, timegrid * 0 + np.squeeze(s.ustar), 'r--')
  axs[0].set_xlabel('Time steps')
  axs[0].set_ylabel('Values')
  axs[0].legend()
  axs[0].grid(True)

  axs[1].plot(timegrid, states[:, 0], label='Position')
  axs[1].plot(timegrid, states[:, 0] * events[:, 2], marker='o', markerfacecolor='none', linestyle='None')
  axs[1].plot(timegrid, timegrid * 0 + s.xstar[0], 'r--')
  axs[1].set_xlabel('Time steps')
  axs[1].set_ylabel('Values')
  axs[1].legend()
  axs[1].grid(True)

  axs[2].plot(timegrid, states[:, 1], label='Velocity')
  axs[2].plot(timegrid, states[:, 1] * events[:, 2], marker='o', markerfacecolor='none', linestyle='None')
  axs[2].plot(timegrid, timegrid * 0 + s.xstar[1], 'r--')
  axs[2].set_xlabel('Time steps')
  axs[2].set_ylabel('Values')
  axs[2].legend()
  axs[2].grid(True)

  axs[3].plot(timegrid, states[:, 2], label='Integrator state')
  axs[3].plot(timegrid, states[:, 2] * events[:, 2], marker='o', markerfacecolor='none', linestyle='None')
  axs[3].plot(timegrid, timegrid * 0 + s.xstar[2], 'r--')
  axs[3].set_xlabel('Time steps')
  axs[3].set_ylabel('Values')
  axs[3].legend()
  axs[3].grid(True)
  plt.show()

  plt.plot(timegrid, etas[:, 0], label='Eta_1')
  plt.plot(timegrid, etas[:, 1], label='Eta_2')
  plt.plot(timegrid, etas[:, 2], label='Eta_3')
  plt.legend()
  plt.grid(True)
  plt.show()

  plt.plot(timegrid, lyap, label='Lyapunov function')
  plt.legend()
  plt.grid(True)
  plt.show()

    # from systems_and_LMI.systems.NonLinPendulum_train import NonLinPendulum_train
    # import matplotlib.pyplot as plt

    # s = NonLinPendulum_train(W, b, 0.0)

    # ref_bound = 5 * np.pi / 180
    # in_ellip = False
    # while not in_ellip:
    #   theta = np.random.uniform(-np.pi/2, np.pi/2)
    #   vtheta = np.random.uniform(-s.max_speed, s.max_speed)
    #   x0 = np.array([[theta], [vtheta], [0.0]])
    #   ref = np.random.uniform(-ref_bound, ref_bound)
    #   s = NonLinPendulum_train(W, b, ref)
    #   if (x0).T @ P @ (x0) <= 1.0:
    #     in_ellip = True
    #     print(f"Initial state: theta0 = {theta*180/np.pi:.2f} deg, vtheta0 = {vtheta:.2f} rad/s, constant reference = {ref*180/np.pi:.2f} deg")
    #     s.state = x0
    
    # nsteps = 500

    # states = []
    # inputs = []
    # lyap = []

    # for i in range(nsteps):
    #   state, u = s.step()
    #   states.append(state)
    #   inputs.append(u)
    #   lyap.append((state - s.xstar).T @ P @ (state - s.xstar))

    # states = np.insert(states, 0, x0, axis=0)
    # states = np.delete(states, -1, axis=0)
    # states = np.squeeze(np.array(states))
    # states[:, 0] *= 180 / np.pi
    # s.xstar[0] *= 180 / np.pi

    # inputs = np.insert(inputs, 0, np.array(0.0), axis=0)
    # inputs = np.delete(inputs, -1, axis=0)
    # inputs = np.squeeze(np.array(inputs))*s.max_torque

    # lyap = np.squeeze(np.array(lyap))

    # timegrid = np.arange(0, nsteps)

        
    # fig, axs = plt.subplots(4, 1)
    # axs[0].plot(timegrid, inputs, label='Control input')
    # axs[0].plot(timegrid, timegrid * 0 + s.wstar[-1] * s.max_torque, 'r--')
    # axs[0].set_xlabel('Time steps')
    # axs[0].set_ylabel('Values')
    # axs[0].legend()
    # axs[0].grid(True)

    # axs[1].plot(timegrid, states[:, 0], label='Position')
    # axs[1].plot(timegrid, timegrid * 0 + s.xstar[0], 'r--')
    # axs[1].set_xlabel('Time steps')
    # axs[1].set_ylabel('Values')
    # axs[1].legend()
    # axs[1].grid(True)

    # axs[2].plot(timegrid, states[:, 1], label='Velocity')
    # axs[2].plot(timegrid, timegrid * 0 + s.xstar[1], 'r--')
    # axs[2].set_xlabel('Time steps')
    # axs[2].set_ylabel('Values')
    # axs[2].legend()
    # axs[2].grid(True)

    # axs[3].plot(timegrid, states[:, 2], label='Integrator state')
    # axs[3].plot(timegrid, timegrid * 0 + s.xstar[2], 'r--')
    # axs[3].set_xlabel('Time steps')
    # axs[3].set_ylabel('Values')
    # axs[3].legend()
    # axs[3].grid(True)
    # plt.show()

    # plt.plot(timegrid, lyap, label='Lyapunov function')
    # plt.legend()
    # plt.grid(True)
    # plt.show()