from system import System
import numpy as np
import cvxpy as cp
import params
import os
import warnings

class LMI_Finsler():
  
  def __init__(self):
    self.system = System()
    self.A = self.system.A
    self.B = self.system.B
    self.nx = self.system.nx
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
    self.neurons = [32, 32]
    self.nlayers = 3
    self.gammas = params.gammas
    self.gamma1 = self.gammas[0]
    self.gamma2 = self.gammas[1]
    self.eta0 = params.eta0
    self.nbigx = self.nx + self.neurons[0] * 2
    
    # Constraint related parameters
    self.m_thresh = 1e-6
    
    # Variables definition
    self.P = cp.Variable((self.nx, self.nx), symmetric=True)
    T_val = cp.Variable(self.nphi)
    self.T = cp.diag(T_val)
    self.T1 = self.T[:self.neurons[0], :self.neurons[0]]
    self.T2 = self.T[self.neurons[0]:, self.neurons[0]:]
    
    self.Z1 = cp.Variable((self.neurons[0], self.nx))
    self.Z2 = cp.Variable((self.neurons[1], self.nx))
    self.Z = cp.vstack([self.Z1, self.Z2])

    # self.bigX1 = cp.Variable((self.nbigx, self.nbigx))
    # self.bigX2 = cp.Variable((self.nbigx, self.nbigx))
    # self.bigX = cp.bmat([
    #   [self.bigX1, np.zeros((self.bigX1.shape[0], self.bigX2.shape[1]))],
    #   [np.zeros((self.bigX2.shape[0], self.bigX1.shape[1])), self.bigX2]
    # ])

    # self.N1 = cp.Variable((self.nx, self.nphi))
    # self.N2 = cp.Variable((self.nphi, self.nphi))
    # self.N3 = cp.Variable((self.nphi, self.nphi))
    # self.N = cp.vstack([self.N1, self.N2, self.N3])

    # Auxiliary parameters
    self.Abar = self.A + self.B @ self.Rw
    self.Bbar = -self.B @ self.Nuw @ self.R
    # self.bigXbar = cp.bmat([
    #   [self.gamma1 * self.bigX1, np.zeros((self.bigX1.shape[0], self.bigX2.shape[1]))],
    #   [np.zeros((self.bigX2.shape[0], self.bigX1.shape[1])), self.gamma2 * self.bigX2]
    # ])
    
    idx = np.eye(self.nx) / (self.nlayers - 1) 
    xzero = np.zeros((self.nx, self.neurons[0]))

    id = np.eye(self.neurons[0])
    zero = np.zeros((self.neurons[0], self.neurons[0]))
    zerox = np.zeros((self.neurons[0], self.nx))

    self.Rxi = np.block([
      [idx, xzero, xzero, xzero, xzero],
      [zerox, id, zero, zero, zero],
      [zerox, zero, zero, id, zero],
      [idx, xzero, xzero, xzero, xzero],
      [zerox, zero, id, zero, zero],
      [zerox, zero, zero, zero, id]
    ])

    # self.Omega1 = cp.bmat([
    #   [np.zeros((self.nx, self.nx)), np.zeros((self.nx, self.neurons[0])), np.zeros((self.nx, self.neurons[0]))],
    #   [self.Z1, self.T1, -self.T1],
    #   [np.zeros((self.neurons[0], self.nx)), np.zeros((self.neurons[0], self.neurons[0])), np.zeros((self.neurons[0], self.neurons[0]))]
    # ])
    # self.Omega2 = cp.bmat([
    #   [np.zeros((self.nx, self.nx)), np.zeros((self.nx, self.neurons[1])), np.zeros((self.nx, self.neurons[1]))],
    #   [self.Z2, self.T2, -self.T2],
    #   [np.zeros((self.neurons[1], self.nx)), np.zeros((self.neurons[1], self.neurons[1])), np.zeros((self.neurons[1], self.neurons[1]))]
    # ])
    # self.Omega = cp.bmat([
    #   [self.Omega1, np.zeros((self.Omega1.shape[0], self.Omega2.shape[1]))],
    #   [np.zeros((self.Omega2.shape[0], self.Omega1.shape[1])), self.Omega2]
    # ])

    self.Rnu = cp.bmat([
      [np.eye(self.nx), np.zeros((self.nx, self.nphi))],
      [np.zeros((self.nphi, self.nx)), np.eye(self.nphi)],
      [self.R @ self.Nvx, np.eye(self.R.shape[0]) - self.R]
    ])
      
    # Parameters definition
    self.alpha = cp.Parameter(nonneg=True)
    
    # Constrain matrices definition
    # hconstr = cp.hstack([self.R @ self.Nvx, np.eye(self.R.shape[0]) - self.R, -np.eye(self.nphi)]) 

    # newconstr = self.Rxi.T @ (self.bigX - self.Omega + self.bigX.T - self.Omega.T) @ self.Rxi + self.N @ hconstr + hconstr.T @ self.N.T
    # T_mat_name = os.path.abspath(__file__ + "/../mat-weights/T_try.npy")
    # Z_mat_name = os.path.abspath(__file__ + "/../mat-weights/Z_try.npy")
    # T = np.load(T_mat_name)
    # Z = np.load(Z_mat_name)
    # self.Tlist = T
    # self.Zlist = Z
    # self.Z = np.split(Z, [32, 64])
    # self.T = []
    # neurons = [32, 32]
    # for i in range(2):
    #   self.T.append(T[i*neurons[i]:(i+1)*neurons[i], i*neurons[i]:(i+1)*neurons[i]])
    self.Omega1 = cp.bmat([
      [np.zeros((self.nx, self.nx)), np.zeros((self.nx, self.neurons[0])), np.zeros((self.nx, self.neurons[0]))],
      [self.gamma1 * self.Z1, self.gamma1 * self.T1, -self.gamma1* self.T1],
      [np.zeros((self.neurons[0], self.nx)), np.zeros((self.neurons[0], self.neurons[0])), np.zeros((self.neurons[0], self.neurons[0]))]
    ])
    self.Omega2 = cp.bmat([
      [np.zeros((self.nx, self.nx)), np.zeros((self.nx, self.neurons[1])), np.zeros((self.nx, self.neurons[1]))],
      [self.gamma2 * self.Z2, self.gamma2 * self.T2, -self.gamma2 * self.T2],
      [np.zeros((self.neurons[1], self.nx)), np.zeros((self.neurons[1], self.neurons[1])), np.zeros((self.neurons[1], self.neurons[1]))]
    ])
    self.Omega = cp.bmat([
      [self.Omega1, np.zeros((self.Omega1.shape[0], self.Omega2.shape[1]))],
      [np.zeros((self.Omega2.shape[0], self.Omega1.shape[1])), self.Omega2]
    ])
    # newconstr = self.Rxi.T @ (self.bigX - self.Omega + self.bigX.T - self.Omega.T) @ self.Rxi

    Rphi = cp.bmat([
        [np.eye(self.nx), np.zeros((self.nx, self.nphi))],
        [self.R @ self.Nvx, np.eye(self.nphi) - self.R],
        [np.zeros((self.nphi, self.nx)), np.eye(self.nphi)]
    ])
    
    self.Rnu = cp.bmat([
      [np.eye(self.nx), np.zeros((self.nx, self.nphi))],
      [np.zeros((self.nphi, self.nx)), np.eye(self.nphi)],
      [self.R @ self.Nvx, np.eye(self.R.shape[0]) - self.R]
    ])

    gamma1 = np.ones(32)*self.gamma1

    gamma2 = np.ones(32)*self.gamma2

    gammavec = np.concatenate([gamma1, gamma2], axis=0)

    gamma = cp.diag(gammavec)

    mat = cp.bmat([
        [np.zeros((self.nx, self.nx)), np.zeros((self.nx, self.nphi)), np.zeros((self.nx, self.nphi))],
        [gamma @ self.Z, -gamma @ self.T, gamma @ self.T]
    ])

    self.M = cp.bmat([
      [self.Abar.T @ self.P @ self.Abar - self.P, self.Abar.T @ self.P @ self.Bbar],
      [self.Bbar.T @ self.P @ self.Abar, self.Bbar.T @ self.P @ self.Bbar]
    ]) + self.Rnu.T @ self.Rxi.T @ (self.Omega + self.Omega.T) @ self.Rxi @ self.Rnu
    # ]) + Rphi.T @ mat.T + mat @ Rphi

    # Constraints definiton
    self.constraints = [self.P >> 0]
    self.constraints += [self.T >> 0]
    # self.constraints += [newconstr << 0]
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
    np.save(f"{path_dir}/bigX1.npy", self.bigX1.value)
    np.save(f"{path_dir}/bigX2.npy", self.bigX2.value)
    np.save(f"{path_dir}/P.npy", self.P.value)
    np.save(f"{path_dir}/T.npy", self.T.value)
    np.save(f"{path_dir}/Z.npy", self.Z.value)
    return self.bigX1.value, self.bigX2.value, self.P.value, self.T.value, self.Z.value

if __name__ == "__main__":
  lmi = LMI_Finsler()
  # alpha = lmi.search_alpha(1, 0, 1e-5, verbose=True)
  alpha = 9 * 1e-4
  lmi.solve(alpha, verbose=True)
  # lmi.save_results('Test')