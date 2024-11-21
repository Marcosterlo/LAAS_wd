from systems_and_LMI.LMI.int_2l.main import LMI_2l_int
import systems_and_LMI.systems.nonlin_2l_xETM.params as params
import numpy as np
import cvxpy as cp
import os
import warnings

class LMI_2l_int_xETM(LMI_2l_int):

  def __init__(self, W, b):
    super().__init__(W, b)
    
    self.neurons = [16, 16, 1]

    gammavec = np.concatenate([params.gammas[i] * np.ones(self.neurons[i]) for i in range(self.nlayers)])
    self.gamma = cp.diag(gammavec)
    self.gamma1 = self.gamma[:self.neurons[0], :self.neurons[0]]
    self.gamma2 = self.gamma[self.neurons[0]:self.neurons[0] + self.neurons[1], self.neurons[0]:self.neurons[0] + self.neurons[1]]
    self.gamma3 = self.gamma[self.neurons[0] + self.neurons[1]:, self.neurons[0] + self.neurons[1]:]

    self.neurons = [16, 16]

    # New variables
    self.nbigx1 = self.neurons[0] * 2 + self.nx
    self.bigX1 = cp.Variable((self.nbigx1, self.nbigx1))
    self.nbigx2 = self.neurons[1] * 2 + self.nx
    self.bigX2 = cp.Variable((self.nbigx2, self.nbigx2))
    self.nbigx3 = 2 + self.nx
    self.bigX3 = cp.Variable((self.nbigx3, self.nbigx3))
    
    self.bigX = cp.bmat([
      [self.bigX1, np.zeros((self.nbigx1, self.nbigx2)), np.zeros((self.nbigx1, self.nbigx3))],
      [np.zeros((self.nbigx2, self.nbigx1)), self.bigX2, np.zeros((self.nbigx2, self.nbigx3))],
      [np.zeros((self.nbigx3, self.nbigx1)), np.zeros((self.nbigx3, self.nbigx2)), self.bigX3]
    ])
    
    self.N1 = cp.Variable((self.nx, self.nphi))
    self.N2 = cp.Variable((self.nphi, self.nphi))
    self.N3 = cp.Variable((self.nphi, self.nphi))
    self.N = cp.vstack([self.N1, self.N2, self.N3])

    self.Z1 = self.Z[:self.neurons[0]]
    self.Z2 = self.Z[self.neurons[0]:self.neurons[0] + self.neurons[1]]
    self.Z3 = self.Z[self.neurons[0] + self.neurons[1]:]
    
    self.T1 = self.T[:self.neurons[0], :self.neurons[0]]
    self.T2 = self.T[self.neurons[0]:self.neurons[0] + self.neurons[1], self.neurons[0]:self.neurons[0] + self.neurons[1]]
    self.T3 = self.T[self.neurons[0] + self.neurons[1]:, self.neurons[0] + self.neurons[1]:]

    # Parameters definition
    self.alpha = cp.Parameter(nonneg=True)

    # Constraint matrices definition

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

    self.Omega1 = cp.bmat([
      [np.zeros((self.nx, self.nx)), np.zeros((self.nx, self.neurons[0])), np.zeros((self.nx, self.neurons[0]))],
      [self.gamma1 @ self.Z1, self.gamma1 @ self.T1, -self.gamma1 @ self.T1],
      [np.zeros((self.neurons[0], self.nx)), np.zeros((self.neurons[0], self.neurons[0])), np.zeros((self.neurons[0], self.neurons[0]))]
    ])

    self.Omega2 = cp.bmat([
      [np.zeros((self.nx, self.nx)), np.zeros((self.nx, self.neurons[1])), np.zeros((self.nx, self.neurons[1]))],
      [self.gamma2 @ self.Z2, self.gamma2 @ self.T2, -self.gamma2 @ self.T2],
      [np.zeros((self.neurons[1], self.nx)), np.zeros((self.neurons[1], self.neurons[1])), np.zeros((self.neurons[1], self.neurons[1]))]
    ])

    self.Omega3 = cp.bmat([
      [np.zeros((self.nx, self.nx)), np.zeros((self.nx, 1)), np.zeros((self.nx, 1))],
      [self.gamma3 * self.Z3, self.gamma3 * self.T3, -self.gamma3 * self.T3],
      [np.zeros((1, self.nx)), np.zeros((1, 1)), np.zeros((1, 1))]
    ])

    self.Omega = cp.bmat([
      [self.Omega1, np.zeros((self.Omega1.shape[0], self.Omega2.shape[1])), np.zeros((self.Omega1.shape[0], self.Omega3.shape[1]))],
      [np.zeros((self.Omega2.shape[0], self.Omega1.shape[1])), self.Omega2, np.zeros((self.Omega2.shape[0], self.Omega3.shape[1]))],
      [np.zeros((self.Omega3.shape[0], self.Omega1.shape[1])), np.zeros((self.Omega3.shape[0], self.Omega2.shape[1])), self.Omega3]
    ])

    self.bigXbar = cp.bmat([
      [self.bigX, np.zeros((self.bigX.shape[0], self.nq))],
      [np.zeros((self.nq, self.bigX.shape[1])), np.zeros((self.nq, self.nq))]
    ])

    idx = np.eye(self.nx)
    xzero = np.zeros((self.nx, self.neurons[0])) 
    xzeros = np.zeros((self.nx, 1))
    xzerox = np.zeros((self.nx, self.nx))

    id = np.eye(self.neurons[0])
    zerox = np.zeros((self.neurons[0], self.nx))
    zero = np.zeros((self.neurons[0], self.neurons[0]))
    zeros = np.zeros((self.neurons[0], 1))

    szerox = np.zeros((1, self.nx))
    szero = np.zeros((1, self.neurons[0]))
    szeros = np.zeros((1, 1))
    ids = np.eye(1)

    self.Rxi = cp.bmat([
      [idx, xzero, xzero, xzeros, xzero, xzero, xzeros],
      [zerox, id, zero, zeros, zero, zero, zeros],
      [zerox, zero, zero, zeros, id, zero, zeros],
      [xzerox, xzero, xzero, xzeros, xzero, xzero, xzeros],
      [zerox, zero, id, zeros, zero, zero, zeros], 
      [zerox, zero, zero, zeros, zero, id, zeros], 
      [xzerox, xzero, xzero, xzeros, xzero, xzero, xzeros],
      [szerox, szero, szero, ids, szero, szero, szeros],
      [szerox, szero, szero, szeros, szero, szero, ids]
    ])

    self.RxiDeltaV = cp.bmat([
      [self.Rxi, np.zeros((self.Rxi.shape[0], self.nq))],
      [np.zeros((self.nq, self.Rxi.shape[1])), np.zeros((self.nq, self.nq))]
    ])


    self.Rnu = cp.bmat([
      [np.eye(self.nx), np.zeros((self.nx, self.nphi)), np.zeros((self.nx, self.nq))],
      [np.zeros((self.nphi, self.nx)), np.eye(self.nphi), np.zeros((self.nphi, self.nq))],
      [self.R @ self.Nvx, np.eye(self.R.shape[0]) - self.R, np.zeros((self.nphi, self.nq))],
      [np.zeros((self.nq, self.nx)), np.zeros((self.nq, self.nphi)), np.eye(self.nq)]
    ])

      
    self.hconstr = cp.hstack([self.R @ self.Nvx, np.eye(self.R.shape[0]) - self.R, -np.eye(self.nphi)])

    self.newconstr = self.Rxi.T @ (self.bigX - self.Omega + self.bigX.T - self.Omega.T) @ self.Rxi + self.N @ self.hconstr + self.hconstr.T @ self.N.T

    self.M = cp.bmat([
      [self.Abar.T @ self.P @ self.Abar - self.P, self.Abar.T @ self.P @ self.Bbar, self.Abar.T @ self.P @ self.C],
      [self.Bbar.T @ self.P @ self.Abar, self.Bbar.T @ self.P @ self.Bbar, self.Bbar.T @ self.P @ self.C],
      [self.C.T @ self.P @ self.Abar, self.C.T @ self.P @ self.Bbar, self.C.T @ self.P @ self.C]
    ]) - self.M1 @ self.Rphi - self.Rphi.T @ self.M1.T + self.Rs.T @ self.Sinsec @ self.Rs - self.Rnu.T @ self.RxiDeltaV.T @ (self.bigXbar + self.bigXbar.T) @ self.RxiDeltaV @ self.Rnu

    # Constraints definition
    self.constraints = [self.P >> 0]
    self.constraints += [self.T >> 0]
    self.constraints += [self.M << -self.m_thresh * np.eye(self.M.shape[0])]
    self.constraints += [self.newconstr << 0]
    
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
  W1_name = os.path.abspath(__file__ + "/../../../systems/nonlin_2l/l1.weight.csv")
  W2_name = os.path.abspath(__file__ + "/../../../systems/nonlin_2l/l2.weight.csv")
  W3_name = os.path.abspath(__file__ + "/../../../systems/nonlin_2l/l3.weight.csv")
  
  b1_name = os.path.abspath(__file__ + "/../../../systems/nonlin_2l/l1.bias.csv")
  b2_name = os.path.abspath(__file__ + "/../../../systems/nonlin_2l/l2.bias.csv")
  b3_name = os.path.abspath(__file__ + "/../../../systems/nonlin_2l/l3.bias.csv")

  W1 = np.loadtxt(W1_name, delimiter=',')
  W2 = np.loadtxt(W2_name, delimiter=',')
  W3 = np.loadtxt(W3_name, delimiter=',')
  W3 = W3.reshape((1, len(W3)))

  W = [W1, W2, W3]

  b1 = np.loadtxt(b1_name, delimiter=',')
  b2 = np.loadtxt(b2_name, delimiter=',')
  b3 = np.loadtxt(b3_name, delimiter=',')
  
  b = [b1, b2, b3] 

  lmi = LMI_2l_int_xETM(W, b)
  # A good alpha value found after a run is 0.125
  # alpha = lmi.search_alpha(1.0, 0.0, 1e-5, verbose=True)
  lmi.solve(0.125, verbose=True)