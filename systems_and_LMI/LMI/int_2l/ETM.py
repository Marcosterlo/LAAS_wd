from systems_and_LMI.LMI.int_2l.main import LMI_2l_int
import systems_and_LMI.systems.nonlin_2l_ETM.params as params
import numpy as np
import cvxpy as cp
import warnings

class LMI_2l_int_ETM(LMI_2l_int):
  
  def __init__(self, W, b):
    super().__init__(W, b)

    self.neurons = [16, 16, 1]

    gammavec = np.concatenate([params.gammas[i] * np.ones(self.neurons[i]) for i in range(self.nlayers)])
    gamma = cp.diag(gammavec)

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

  W1_name = os.path.abspath(__file__ + "/../2l/l1.weight.csv")
  W2_name = os.path.abspath(__file__ + "/../2l/l2.weight.csv")
  W3_name = os.path.abspath(__file__ + "/../2l/l3.weight.csv")

  b1_name = os.path.abspath(__file__ + "/../2l/l1.bias.csv")
  b2_name = os.path.abspath(__file__ + "/../2l/l2.bias.csv")
  b3_name = os.path.abspath(__file__ + "/../2l/l3.bias.csv")

    
  W1 = np.loadtxt(W1_name, delimiter=',')
  W2 = np.loadtxt(W2_name, delimiter=',')
  W3 = np.loadtxt(W3_name, delimiter=',')
  W3 = W3.reshape((1, len(W3)))

  W = [W1, W2, W3]

  b1 = np.loadtxt(b1_name, delimiter=',')
  b2 = np.loadtxt(b2_name, delimiter=',')
  b3 = np.loadtxt(b3_name, delimiter=',')
  
  b = [b1, b2, b3] 

  lmi = LMI_2l_int_ETM(W, b)
  # alpha = lmi.search_alpha(1, 0, 1e-5, verbose=True)
  alpha = 0.1
  lmi.solve(alpha, verbose=True)
  lmi.save_results('res_ETM')