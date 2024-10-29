from systems_and_LMI.systems.NonLinearPendulum import NonLinPendulum
import numpy as np
import cvxpy as cp
import os

class LMI_1l():
  
  def __init__(self):
    self.system = NonLinPendulum()
    self.K = self.system.K
    self.A = self.system.A
    self.B = self.system.B
    self.C = self.system.C
    self.D = self.system.D
    self.nx = self.system.nx
    self.nphi = 1
    self.vbar = self.system.max_torque
    self.xstar = self.system.xstar 
    self.Nux = np.zeros((self.nphi, self.nx))
    self.Nuw = np.array([[1.0]])
    self.Nvx = self.K
    self.Nvw = np.array([[0.0]])
    
    # Auxiliary parameters
    self.R = np.linalg.inv(np.eye(self.Nvw.shape[0]) - self.Nvw)
    self.Rw = self.Nux + self.Nuw @ self.R @ self.Nvx
    self.Abar = self.A + self.B @ self.Rw
    self.Bbar = -self.B @ self.Nuw @ self.R
    
    # Variables definition
    self.P = cp.Variable((self.nx, self.nx), symmetric=True)
    self.T = cp.Variable((self.nphi, self.nphi))
    self.Z = cp.Variable((self.nphi, self.nx))
    
    # Parameters definition
    self.alpha = cp.Parameter()
    
    # Constraint matrices definition
    self.Rphi = cp.bmat([
        [np.eye(self.nx), np.zeros((self.nx, self.nphi)), np.zeros((self.nx, self.nphi))],
        [self.R @ self.Nvx, np.eye(self.R.shape[0]) - self.R, np.array([[0.0]])],
        [np.zeros((self.nphi, self.nx)), np.eye(self.nphi), np.array([[0.0]])],
        [np.zeros((self.nphi, self.nx)), np.zeros((self.nphi, self.nphi)), np.array([[1.0]])]
    ])

    self.M1 = cp.bmat([
      [np.zeros((self.nx, self.nx)), np.zeros((self.nx, self.nphi)), np.zeros((self.nx, self.nphi)), np.zeros((self.nx, self.nphi))],
      [self.Z, -self.T , self.T, np.zeros((self.nphi, self.nphi))],
      [np.zeros((self.nphi, self.nx)), np.zeros((self.nphi, self.nphi)), np.zeros((self.nphi, self.nphi)), np.zeros((self.nphi, self.nphi))]
    ])
    
    self.Sinsec = cp.bmat([
      [0.0, -1.0],
      [-1.0, -2.0]
    ])

    self.Rs = cp.bmat([
      [np.array([[1.0, 0.0, 0.0]]), np.zeros((1, self.nphi)), np.zeros((1, self.nphi))],
      [np.zeros((self.nphi, self.nx)), np.zeros((self.nphi, self.nphi)), np.array([[1.0]])],
    ])
    
    self.ellip = cp.bmat([
      [self.P, self.Z.T],
      [self.Z, 2 * self.alpha * self.T - self.alpha**2 * self.vbar**(-2)]
    ]) 

    self.M = cp.bmat([
      [self.Abar.T @ self.P @ self.Abar - self.P, self.Abar.T @ self.P @ self.Bbar, self.Abar.T @ self.P @ self.C],
      [self.Bbar.T @ self.P @ self.Abar, self.Bbar.T @ self.P @ self.Bbar - self.T, self.Bbar.T @ self.P @ self.C],
      [self.C.T @ self.P @ self.Abar, self.C.T @ self.P @ self.Bbar, self.C.T @ self.P @ self.C]
    ]) - self.M1 @ self.Rphi - self.Rphi.T @ self.M1.T + self.Rs.T @ self.Sinsec @ self.Rs
   
   # Constraints definition
    self.m_thresh = 1e-6
    self.constraints = [self.P >> 0, self.T >> 0, self.ellip >> 0, self.M << -self.m_thresh*np.eye(self.M.shape[0])]

    # Objective function definition
    self.objective = cp.Minimize(cp.trace(self.P))

    # Problem definition
    self.prob = cp.Problem(self.objective, self.constraints)
  
  def solve(self, alpha_val, verbose=False):
    self.alpha.value = alpha_val
    self.prob.solve(solver=cp.MOSEK, verbose=False)

    if self.prob.status not in ["optimal", "optimal_inaccurate"]:
      return None
    else:
      if verbose:
        print(f"Max eigenvalue of P: {np.max(np.linalg.eigvals(self.P.value))}")
        print(f"Max eigenvalue of M: {np.max(np.linalg.eigvals(self.M.value))}")
      return self.P, self.T, self.Z
  
  def search_alpha(self, feasible_extreme, infeasible_extreme, threshold, verbose=False):
    golden_ratio = (1 + np.sqrt(5)) / 2
    i = 0
    while (feasible_extreme - infeasible_extreme > threshold):
      i += 1
      alpha1 = feasible_extreme - (feasible_extreme - infeasible_extreme) / golden_ratio
      alpha2 = infeasible_extreme + (feasible_extreme - infeasible_extreme) / golden_ratio

      P1, _, _ = self.solve(alpha1, verbose=False)
      if P1 is None:
        val1 = 1e10
      else:
        val1 = np.max(np.linalg.eigvals(P1.value))

      P2, _, _ = self.solve(alpha2, verbose=False)
      if P2 is None:
        val2 = 1e10
      else:
        val2 = np.max(np.linalg.eigvals(P2.value))

      if val1 < val2:
        feasible_extreme = alpha2
      else:
        infeasible_extreme = alpha1

      if verbose:
        print(f"\nIteration number: {i}")
        print(f"Current alpha value: {feasible_extreme}")

    return feasible_extreme
  
  def save_results(self, path_dir: str):
    if not os.path.exists(path_dir):
      os.makedirs(path_dir)
    P, T, Z = self.solve(self.alpha.value)
    np.save(f"{path_dir}/P.npy", P.value)
    np.save(f"{path_dir}/T.npy", T.value)
    np.save(f"{path_dir}/Z.npy", Z.value)
    return P, T, Z

if __name__ == "__main__":
  lmi = LMI_1l()
  alpha = lmi.alpha.value = lmi.search_alpha(1.0, 0.0, 1e-6, verbose=True)
  lmi.solve(alpha, verbose=True)
  lmi.save_results('Test')