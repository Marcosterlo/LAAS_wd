from systems_and_LMI.systems.NonLinPendulum_no_int_train import NonLinPendulum_no_int_train
import numpy as np
import cvxpy as cp
import os
from scipy.linalg import block_diag

class LMI_3l_no_int():
  def __init__(self, W, b):

    self.system = NonLinPendulum_no_int_train(W, b)
    self.A = self.system.A
    self.B = self.system.B
    self.C = self.system.C
    self.nx = self.system.nx
    self.nq = self.system.nq
    self.bound = 1
    self.max_torque = self.system.max_torque
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
    self.neurons = self.system.neurons
    self.nlayers = self.system.nlayers

    # Constraint related parameters
    self.m_thresh = 1e-6
    
    # Auxiliary parameters
    self.Abar = self.A + self.B @ self.Rw
    self.Bbar = -self.B @ self.Nuw @ self.R
    
    # Variables definition
    self.P = cp.Variable((self.nx, self.nx), symmetric=True)
    T_val = cp.Variable(self.nphi)
    self.T = cp.diag(T_val)

    Z1 = cp.Variable((self.neurons[0], self.nx))
    Z2 = cp.Variable((self.neurons[1], self.nx))
    Z3 = cp.Variable((self.neurons[2], self.nx))
    Z4 = cp.Variable((1, self.nx))
    self.Z = cp.vstack([Z1, Z2, Z3, Z4])
    
    # Parameters definition
    self.alpha = cp.Parameter()

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
      [np.array([[1.0, 0.0]]), np.zeros((1, self.nphi)), np.zeros((1, self.nq))],
      [np.zeros((self.nq, self.nx)), np.zeros((1, self.nphi)), np.eye(self.nq)]
    ])

    self.M = cp.bmat([
      [self.Abar.T @ self.P @ self.Abar - self.P, self.Abar.T @ self.P @ self.Bbar, self.Abar.T @ self.P @ self.C],
      [self.Bbar.T @ self.P @ self.Abar, self.Bbar.T @ self.P @ self.Bbar, self.Bbar.T @ self.P @ self.C],
      [self.C.T @ self.P @ self.Abar, self.C.T @ self.P @ self.Bbar, self.C.T @ self.P @ self.C]
    ]) - self.M1 @ self.Rphi - self.Rphi.T @ self.M1.T + self.Rs.T @ self.Sinsec @ self.Rs

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
    vcap = self.max_torque
    vcap = np.min([np.abs(-self.max_torque - self.wstar[-1]), np.abs(self.max_torque - self.wstar[-1])], axis=0)
    ellip = cp.bmat([
        [self.P, cp.reshape(Z_el, (self.nx ,1))],
        [cp.reshape(Z_el, (1, self.nx)), cp.reshape(2*self.alpha*T_el - self.alpha**2*vcap**(-2), (1, 1))] 
    ])
    self.constraints += [ellip >> 0]
    
    
    # Objective function definition
    self.objective = cp.Minimize(cp.trace(self.P))

    # Problem definition
    self.prob = cp.Problem(self.objective, self.constraints)
  
  def solve(self, alpha_val, verbose=False):
    self.alpha.value = alpha_val
    try:
      self.prob.solve(solver=cp.MOSEK, verbose=False)
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
  W1_name = os.path.abspath(__file__ + "/../simple_weights/l1.weight.csv")
  W2_name = os.path.abspath(__file__ + "/../simple_weights/l2.weight.csv")
  W3_name = os.path.abspath(__file__ + "/../simple_weights/l3.weight.csv")
  W4_name = os.path.abspath(__file__ + "/../simple_weights/l4.weight.csv")

  W1 = np.loadtxt(W1_name, delimiter=',')
  W2 = np.loadtxt(W2_name, delimiter=',')
  W3 = np.loadtxt(W3_name, delimiter=',')
  W4 = np.loadtxt(W4_name, delimiter=',')
  W4 = W4.reshape((1, len(W4)))

  W = [W1, W2, W3, W4]

  b1_name = os.path.abspath(__file__ + "/../simple_weights/l1.bias.csv")
  b2_name = os.path.abspath(__file__ + "/../simple_weights/l2.bias.csv")
  b3_name = os.path.abspath(__file__ + "/../simple_weights/l3.bias.csv")
  b4_name = os.path.abspath(__file__ + "/../simple_weights/l4.bias.csv")
  
  b1 = np.loadtxt(b1_name, delimiter=',')
  b2 = np.loadtxt(b2_name, delimiter=',')
  b3 = np.loadtxt(b3_name, delimiter=',')
  b4 = np.loadtxt(b4_name, delimiter=',')
  
  b = [b1, b2, b3, b4] 

  lmi = LMI_3l_no_int(W, b)
  alpha = lmi.search_alpha(1, 0, 1e-5, verbose=True)
  lmi.solve(alpha, verbose=True)
  lmi.save_results('Test')