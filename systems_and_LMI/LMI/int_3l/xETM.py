from systems_and_LMI.LMI.int_3l.main import LMI_3l_int
import systems_and_LMI.systems.nonlin_exp_ROA_kETM.params as params
import numpy as np
import cvxpy as cp
import os
import warnings

class LMI_3l_int_xETM(LMI_3l_int):

  def __init__(self, W, b):
    super().__init__(W, b)

    gammavec = np.concatenate([params.gammas[i] * np.ones(self.neurons[i]) for i in range(self.nlayers - 1)] + [np.array([1.0])])
    self.gamma = cp.diag(gammavec)
    self.gamma1 = self.gamma[:self.neurons[0], :self.neurons[0]]
    self.gamma2 = self.gamma[self.neurons[0]:self.neurons[0] + self.neurons[1], self.neurons[0]:self.neurons[0] + self.neurons[1]]
    self.gamma3 = self.gamma[self.neurons[0] + self.neurons[1]:self.neurons[0] + self.neurons[1] + self.neurons[2], self.neurons[0] + self.neurons[1]:self.neurons[0] + self.neurons[1] + self.neurons[2]]

    # New variables
    self.nbigx1 = self.neurons[0] * 2 + self.nx
    self.bigX1 = cp.Variable((self.nbigx1, self.nbigx1))
    self.nbigx2 = self.neurons[1] * 2 + self.nx
    self.bigX2 = cp.Variable((self.nbigx2, self.nbigx2))
    self.nbigx3 = self.neurons[2] * 2 + self.nx
    self.bigX3 = cp.Variable((self.nbigx3, self.nbigx3))
    
    self.bigX = cp.bmat([
      [self.bigX1, np.zeros((self.nbigx1, self.nbigx2)), np.zeros((self.nbigx1, self.nbigx3))],
      [np.zeros((self.nbigx2, self.nbigx1)), self.bigX2, np.zeros((self.nbigx2, self.nbigx3))],
      [np.zeros((self.nbigx3, self.nbigx1)), np.zeros((self.nbigx3, self.nbigx2)), self.bigX3]
    ])
    
    self.N1 = cp.Variable((self.nx, self.nphi-1))
    self.N2 = cp.Variable((self.nphi-1, self.nphi-1))
    self.N3 = cp.Variable((self.nphi-1, self.nphi-1))
    self.N = cp.vstack([self.N1, self.N2, self.N3])

    self.Z1 = self.Z[:self.neurons[0]]
    self.Z2 = self.Z[self.neurons[0]:self.neurons[0] + self.neurons[1]]
    self.Z3 = self.Z[self.neurons[0] + self.neurons[1]:self.neurons[0] + self.neurons[1] + self.neurons[2]]
    
    self.T1 = self.T[:self.neurons[0], :self.neurons[0]]
    self.T2 = self.T[self.neurons[0]:self.neurons[0] + self.neurons[1], self.neurons[0]:self.neurons[0] + self.neurons[1]]
    self.T3 = self.T[self.neurons[0] + self.neurons[1]:self.neurons[0] + self.neurons[1] + self.neurons[2], self.neurons[0] + self.neurons[1]:self.neurons[0] + self.neurons[1] + self.neurons[2]] 

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
      [np.zeros((self.nx, self.nx)), self.Z1.T @ self.gamma1.T, np.zeros((self.nx, self.neurons[0]))],
      [self.gamma1 @ self.Z1, -self.gamma1 @ self.T1 - self.T1.T @ self.gamma1.T, self.gamma1 @ self.T1], 
      [np.zeros((self.neurons[0], self.nx)), self.T1.T @ self.gamma1.T, np.zeros((self.neurons[0], self.neurons[0]))]
    ])

    self.Omega2 = cp.bmat([
      [np.zeros((self.nx, self.nx)), self.Z2.T @ self.gamma2.T, np.zeros((self.nx, self.neurons[1]))],
      [self.gamma2 @ self.Z2, -self.gamma2 @ self.T2 - self.T2.T @ self.gamma2.T, self.gamma2 @ self.T2],
      [np.zeros((self.neurons[1], self.nx)), self.T2.T @ self.gamma2.T, np.zeros((self.neurons[1], self.neurons[1]))]
    ])

    self.Omega3 = cp.bmat([
      [np.zeros((self.nx, self.nx)), self.Z3.T @ self.gamma3.T, np.zeros((self.nx, self.neurons[2]))],
      [self.gamma3 @ self.Z3, -self.gamma3 @ self.T3 - self.T3.T @ self.gamma3.T, self.gamma3 @ self.T3],
      [np.zeros((self.neurons[2], self.nx)), self.T3.T @ self.gamma3.T, np.zeros((self.neurons[2], self.neurons[2]))]
    ])

    self.Omega = cp.bmat([
      [self.Omega1, np.zeros((self.Omega1.shape[0], self.Omega2.shape[1])), np.zeros((self.Omega1.shape[0], self.Omega3.shape[1]))],
      [np.zeros((self.Omega2.shape[0], self.Omega1.shape[1])), self.Omega2, np.zeros((self.Omega2.shape[0], self.Omega3.shape[1]))],
      [np.zeros((self.Omega3.shape[0], self.Omega1.shape[1])), np.zeros((self.Omega3.shape[0], self.Omega2.shape[1])), self.Omega3]
    ])

    self.bigXbar = cp.bmat([
      [self.bigX, np.zeros((self.nbigx1 + self.nbigx2 + self.nbigx3, 3))],
      [np.zeros((3, self.nbigx1 + self.nbigx2 + self.nbigx3)), np.zeros((3, 3))]
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
      [idx, xzero, xzero, xzero, xzero, xzero, xzero],
      [zerox, id, zero, zero, zero, zero, zero],
      [zerox, zero, zero, zero, id, zero, zero],
      [xzerox, xzero, xzero, xzero, xzero, xzero, xzero],
      [zerox, zero, id, zero, zero, zero, zero], 
      [zerox, zero, zero, zero, zero, id, zero], 
      [xzerox, xzero, xzero, xzero, xzero, xzero, xzero],
      [zerox, zero, zero, id, zero, zero, zero], 
      [zerox, zero, zero, zero, zero, zero, id]
    ])
    
    Rxi1 = cp.bmat([
      [idx, xzero, xzero, xzero, xzeros, xzero, xzero, xzero, xzeros],
      [zerox, id, zero, zero, zeros, zero, zero, zero, zeros],
      [zerox, zero, zero, zero, zeros, id, zero, zero, zeros],
      [xzerox, xzero, xzero, xzero, xzeros, xzero, xzero, xzero, xzeros],
      [zerox, zero, id, zero, zeros, zero, zero, zero, zeros], 
      [zerox, zero, zero, zero, zeros, zero, id, zero, zeros], 
      [xzerox, xzero, xzero, xzero, xzeros, xzero, xzero, xzero, xzeros],
      [zerox, zero, zero, id, zeros, zero, zero, zero, zeros], 
      [zerox, zero, zero, zero, zeros, zero, zero, id, zeros],
      [szerox, szero, szero, szero, ids, szero, szero, szero, szeros],
      [szerox, szero, szero, szero, szeros, szero, szero, szero, ids]
    ])

    self.Rxi1 = cp.bmat([
      [Rxi1, np.zeros((Rxi1.shape[0], 1))],
      [np.zeros((1, Rxi1.shape[1])), np.array([[0.0]])]
    ])


    self.Rxi2 = cp.bmat([
        [np.eye(self.nx), np.zeros((self.nx, self.nphi)), np.zeros((self.nx, self.nq))],
        [np.zeros((self.nphi, self.nx)), np.eye(self.nphi), np.zeros((self.nphi, self.nq))],
        [self.R @ self.Nvx, np.eye(self.R.shape[0]) - self.R, np.zeros((self.nphi, self.nq))],
        [np.zeros((self.nq, self.nx)), np.zeros((self.nq, self.nphi)), np.eye(self.nq)],
    ])

    self.Rx = self.Rxi1 @ self.Rxi2
      
    self.hconstr = cp.hstack([(self.R @ self.Nvx)[:-1], (np.eye(self.R.shape[0]) - self.R)[:-1, :-1], -np.eye(self.nphi-1)])

    self.newconstr = self.Rxi.T @ (self.bigX - self.Omega) @ self.Rxi + self.N @ self.hconstr + self.hconstr.T @ self.N.T

    self.M = cp.bmat([
      [self.Abar.T @ self.P @ self.Abar - self.P, self.Abar.T @ self.P @ self.Bbar, self.Abar.T @ self.P @ self.C],
      [self.Bbar.T @ self.P @ self.Abar, self.Bbar.T @ self.P @ self.Bbar, self.Bbar.T @ self.P @ self.C],
      [self.C.T @ self.P @ self.Abar, self.C.T @ self.P @ self.Bbar, self.C.T @ self.P @ self.C]
    ]) - self.M1 @ self.Rphi - self.Rphi.T @ self.M1.T + self.Rs.T @ self.Sinsec @ self.Rs + self.Rx.T @ self.bigXbar @ self.Rx 

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
  W1_name = os.path.abspath(__file__ + "/../../../systems/nonlin_norm_weights/mlp_extractor.policy_net.0.weight.csv")
  W2_name = os.path.abspath(__file__ + "/../../../systems/nonlin_norm_weights/mlp_extractor.policy_net.2.weight.csv")
  W3_name = os.path.abspath(__file__ + "/../../../systems/nonlin_norm_weights/mlp_extractor.policy_net.4.weight.csv")
  W4_name = os.path.abspath(__file__ + "/../../../systems/nonlin_norm_weights/action_net.weight.csv")
  
  b1_name = os.path.abspath(__file__ + "/../../../systems/nonlin_norm_weights/mlp_extractor.policy_net.0.bias.csv")
  b2_name = os.path.abspath(__file__ + "/../../../systems/nonlin_norm_weights/mlp_extractor.policy_net.2.bias.csv")
  b3_name = os.path.abspath(__file__ + "/../../../systems/nonlin_norm_weights/mlp_extractor.policy_net.4.bias.csv")
  b4_name = os.path.abspath(__file__ + "/../../../systems/nonlin_norm_weights/action_net.bias.csv")

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

  lmi = LMI_3l_int_xETM(W, b)
  lmi.solve(1.0, verbose=True)