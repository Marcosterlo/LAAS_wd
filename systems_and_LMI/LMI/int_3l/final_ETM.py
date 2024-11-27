from systems_and_LMI.systems.FinalPendulum import FinalPendulum
import systems_and_LMI.systems.final.params as params
import numpy as np
import cvxpy as cp
import os
import warnings

class LMI_final():
  def __init__(self, W, b):
    
    self.system = FinalPendulum(W, b, 0.0)
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
    self.nlayers = self.system.nlayers
    self.neurons = [32, 32, 32]
    self.gammas = params.gammas
    self.gamma1_scal = self.gammas[0]
    self.gamma2_scal = self.gammas[1]
    self.gamma3_scal = self.gammas[2]
    self.nbigx = self.nx + self.neurons[0] * 2

    # AGGIUNGI PSI_SAT AL VETTORE PER CUI SI DEVE RACCOGLIERE TUTTO: [TILDE X, TILDE PSI, TILDE PSI_SAT, TILDE PHI]
    
#     # Variables definition
#     self.P = cp.Variable((self.nx, self.nx), symmetric=True)

#     # Sector variables
#     T_val = cp.Variable(self.nphi)
#     self.T = cp.diag(T_val)
#     self.T1 = self.T[:self.neurons[0], :self.neurons[0]]
#     self.T2 = self.T[self.neurons[0]:self.neurons[0] + self.neurons[1], self.neurons[0]:self.neurons[0] + self.neurons[1]]
#     self.T3 = self.T[self.neurons[0] + self.neurons[1]:, self.neurons[0] + self.neurons[1]:]

#     self.Z = cp.Variable((self.nphi, self.nx))
#     self.Z1 = self.Z[:self.neurons[0], :]
#     self.Z2 = self.Z[self.neurons[0]:self.neurons[0] + self.neurons[1], :]
#     self.Z3 = self.Z[self.neurons[0] + self.neurons[1]:, :]

#     # New ETM matrices
#     self.bigX1 = cp.Variable((self.nbigx, self.nbigx))
#     self.bigX2 = cp.Variable((self.nbigx, self.nbigx))
#     self.bigX3 = cp.Variable((self.nbigx, self.nbigx))
    
#     # Finsler multipliers
#     self.N11 = cp.Variable((self.nx, self.nphi))
#     self.N12 = cp.Variable((self.nphi, self.nphi), symmetric=True)
#     N13 = cp.Variable(self.nphi)
#     self.N13 = cp.diag(N13)
#     self.N1 = cp.vstack([self.N11, self.N12, self.N13])

#     self.N21 = cp.Variable((self.nx, self.nphi))
#     self.N22 = cp.Variable((self.nphi, self.nphi), symmetric=True)
#     N23 = cp.Variable(self.nphi)
#     self.N23 = cp.diag(N23)
#     self.N2 = cp.vstack([self.N21, self.N22, self.N23])
    
#     self.N31 = cp.Variable((self.nx, self.nphi))
#     self.N32 = cp.Variable((self.nphi, self.nphi), symmetric=True)
#     N33 = cp.Variable(self.nphi)
#     self.N33 = cp.diag(N33)
#     self.N3 = cp.vstack([self.N31, self.N32, self.N33])

#     # Constraint related parameters
#     self.m_thresh = 1e-6
    
#     # Auxiliary parameters
#     self.Abar = self.A + self.B @ self.Rw
#     self.Bbar = -self.B @ self.Nuw @ self.R

#     # Useful variables to build the transformation matrices
#     idx = np.eye(self.nx)
#     xzero = np.zeros((self.nx, self.neurons[0]))

#     id = np.eye(self.neurons[0])
#     zero = np.zeros((self.neurons[0], self.neurons[0]))
#     zerox = np.zeros((self.neurons[0], self.nx))

#     self.R1 = cp.bmat([
#       [idx, xzero, xzero, xzero, xzero, xzero, xzero],
#       [zerox, id, zero, zero, zero, zero, zero],
#       [zerox, zero, zero, zero, id, zero, zero],
#     ])

#     self.R2 = cp.bmat([
#       [idx, xzero, xzero, xzero, xzero, xzero, xzero],
#       [zerox, zero, id, zero, zero, zero, zero],
#       [zerox, zero, zero, zero, zero, id, zero],
#     ])
    
#     self.R3 = cp.bmat([
#       [idx, xzero, xzero, xzero, xzero, xzero, xzero],
#       [zerox, zero, zero, id, zero, zero, zero],
#       [zerox, zero, zero, zero, zero, zero, id],
#     ])

#     # Transformation matrix to pass from xi = [x, psi1, psi2, psi3, nu1, nu2, nu3] to [x, psi1, psi2, psi3]
#     self.Rnu = cp.bmat([
#       [np.eye(self.nx), np.zeros((self.nx, self.nphi)), np.zeros((self.nx, self.nq))],
#       [np.zeros((self.nphi, self.nx)), np.eye(self.nphi), np.zeros((self.nphi, self.nq))],
#       [self.R @ self.Nvx, np.eye(self.R.shape[0]) - self.R, np.zeros((self.nphi, self.nq))],
#     ])
    
#     # Parameters definition
#     self.alpha = cp.Parameter(nonneg=True)
    
#     # Old ETM (and sec condition) structures with respect to vector [x, psi, nu]
#     self.Omega1 = cp.bmat([
#       [np.zeros((self.nx, self.nx)), np.zeros((self.nx, self.neurons[0])), np.zeros((self.nx, self.neurons[0]))],
#       [self.Z1, self.T1, -self.T1],
#       [np.zeros((self.neurons[0], self.nx)), np.zeros((self.neurons[0], self.neurons[0])), np.zeros((self.neurons[0], self.neurons[0]))]
#     ])
    
#     self.Omega2 = cp.bmat([
#       [np.zeros((self.nx, self.nx)), np.zeros((self.nx, self.neurons[1])), np.zeros((self.nx, self.neurons[1]))],
#       [self.Z2, self.T2, -self.T2],
#       [np.zeros((self.neurons[1], self.nx)), np.zeros((self.neurons[1], self.neurons[1])), np.zeros((self.neurons[1], self.neurons[1]))]
#     ])
    
#     self.Omega3 = cp.bmat([
#       [np.zeros((self.nx, self.nx)), np.zeros((self.nx, self.neurons[2])), np.zeros((self.nx, self.neurons[2]))],
#       [self.Z3, self.T3, -self.T3],
#       [np.zeros((self.neurons[2], self.nx)), np.zeros((self.neurons[2], self.neurons[2])), np.zeros((self.neurons[2], self.neurons[2]))]
#     ])
    
#     self.Sinsec = cp.bmat([
#       [0.0, -1.0],
#       [-1.0, -2.0]
#     ])
    
#     self.Rs = cp.bmat([
#       [np.array([[1.0, 0.0, 0.0]]), np.zeros((1, self.nphi)), np.zeros((1, self.nq))],
#       [np.zeros((self.nq, self.nx)), np.zeros((1, self.nphi)), np.eye(self.nq)]
#     ])

#     # Constraint matrices definition
#     # Finsler constraint to handle nu with respect to x and psi
#     self.hconstr = cp.hstack([self.R @ self.Nvx, np.eye(self.R.shape[0]) - self.R, -np.eye(self.nphi)])

#     # Finsler constraints
#     self.finsler1 = self.R1.T @ (self.bigX1 - self.Omega1 + self.bigX1.T - self.Omega1.T) @ self.R1 + self.N1 @ self.hconstr + self.hconstr.T @ self.N1.T

#     self.finsler2 = self.R2.T @ (self.bigX2 - self.Omega2 + self.bigX2.T - self.Omega2.T) @ self.R2 + self.N2 @ self.hconstr + self.hconstr.T @ self.N2.T
    
#     self.finsler3 = self.R3.T @ (self.bigX3 - self.Omega3 + self.bigX3.T - self.Omega3.T) @ self.R3 + self.N3 @ self.hconstr + self.hconstr.T @ self.N3.T

#     self.M = cp.bmat([
#       [self.Abar.T @ self.P @ self.Abar - self.P, self.Abar.T @ self.P @ self.Bbar, self.Abar.T @ self.P @ self.C],
#       [self.Bbar.T @ self.P @ self.Abar, self.Bbar.T @ self.P @ self.Bbar, self.Bbar.T @ self.P @ self.C],
#       [self.C.T @ self.P @ self.Abar, self.C.T @ self.P @ self.Bbar, self.C.T @ self.P @ self.C]
#     ]) + self.Rs.T @ self.Sinsec @ self.Rs + self.Rnu.T @ (self.R1.T @ (self.gamma1_scal * (self.bigX1 + self.bigX1.T)) @ self.R1 + self.R2.T @ (self.gamma2_scal * (self.bigX2 + self.bigX2.T)) @ self.R2 + self.R3.T @ (self.gamma3_scal * (self.bigX3 + self.bigX3.T)) @ self.R3) @ self.Rnu


#     # Constraints definiton
#     self.constraints = [self.P >> 0]
#     self.constraints += [self.T >> 0]
#     self.constraints += [self.finsler1 << 0]
#     self.constraints += [self.finsler2 << 0]
#     self.constraints += [self.finsler3 << 0]
#     self.constraints += [self.M << -self.m_thresh * np.eye(self.M.shape[0])]
    
#     # Ellipsoid conditions for activation functions
#     for i in range(self.nlayers - 1):
#       for k in range(self.neurons[i]):
#         Z_el = self.Z[i*self.neurons[i] + k]
#         T_el = self.T[i*self.neurons[i] + k, i*self.neurons[i] + k]
#         vcap = np.min([np.abs(-self.bound - self.wstar[i][k][0]), np.abs(self.bound - self.wstar[i][k][0])], axis=0)
#         ellip = cp.bmat([
#             [self.P, cp.reshape(Z_el, (self.nx ,1))],
#             [cp.reshape(Z_el, (1, self.nx)), cp.reshape(2*self.alpha*T_el - self.alpha**2*vcap**(-2), (1, 1))] 
#         ])
#         self.constraints += [ellip >> 0]
    
#     # # Ellipsoid conditions for last saturation
#     # Z_el = self.Z[-1, :]
#     # T_el = self.T[-1, -1]
#     # vcap = np.min([np.abs(-self.bound - self.wstar[-1]), np.abs(self.bound - self.wstar[-1])], axis=0)
#     # ellip = cp.bmat([
#     #     [self.P, cp.reshape(Z_el, (self.nx ,1))],
#     #     [cp.reshape(Z_el, (1, self.nx)), cp.reshape(2*self.alpha*T_el - self.alpha**2*vcap**(-2), (1, 1))] 
#     # ])
#     # self.constraints += [ellip >> 0]
    
#     # Objective function definition
#     self.objective = cp.Minimize(cp.trace(self.P))

#     # Problem definition
#     self.prob = cp.Problem(self.objective, self.constraints)

#     # User warnings filter
#     warnings.filterwarnings("ignore", category=UserWarning, module='cvxpy')

#   def solve(self, alpha_val, verbose=False):
#     self.alpha.value = alpha_val
#     try:
#       self.prob.solve(solver=cp.SCS, verbose=True)
#     except cp.error.SolverError:
#       return None, None, None

#     if self.prob.status not in ["optimal", "optimal_inaccurate"]:
#       return None, None, None
#     else:
#       if verbose:
#         print(f"Max eigenvalue of P: {np.max(np.linalg.eigvals(self.P.value))}")
#         print(f"Max eigenvalue of M: {np.max(np.linalg.eigvals(self.M.value))}") 
#       return self.P.value, self.T.value, self.Z.value
  
#   def search_alpha(self, feasible_extreme, infeasible_extreme, threshold, verbose=False):
#     golden_ratio = (1 + np.sqrt(5)) / 2
#     i = 0
#     while (feasible_extreme - infeasible_extreme > threshold) and i < 11:
#       i += 1
#       alpha1 = feasible_extreme - (feasible_extreme - infeasible_extreme) / golden_ratio
#       alpha2 = infeasible_extreme + (feasible_extreme - infeasible_extreme) / golden_ratio
      
#       P1, _, _ = self.solve(alpha1, verbose=False)
#       if P1 is None:
#         val1 = 1e10
#       else:
#         val1 = np.max(np.linalg.eigvals(P1))
      
#       P2, _, _ = self.solve(alpha2, verbose=False)
#       if P2 is None:
#         val2 = 1e10
#       else:
#         val2 = np.max(np.linalg.eigvals(P2))
        
#       if val1 < val2:
#         feasible_extreme = alpha2
#       else:
#         infeasible_extreme = alpha1
        
#       if verbose:
#         if val1 < val2:
#           P_eig = val1
#         else:
#           P_eig = val2
#         print(f"\nIteration number: {i}")
#         print(f"==================== \nMax eigenvalue of P: {P_eig}")
#         print(f"Current alpha value: {feasible_extreme}\n==================== \n")
    
#     return feasible_extreme
  
#   def save_results(self, path_dir: str):
#     if not os.path.exists(path_dir):
#       os.makedirs(path_dir)
#     np.save(f"{path_dir}/bigX1.npy", self.bigX1.value)
#     np.save(f"{path_dir}/bigX2.npy", self.bigX2.value)
#     np.save(f"{path_dir}/bigX3.npy", self.bigX3.value)
#     np.save(f"{path_dir}/P.npy", self.P.value)
#     np.save(f"{path_dir}/T.npy", self.T.value)
#     np.save(f"{path_dir}/Z.npy", self.Z.value)
#     return self.bigX1.value, self.bigX2.value, self.bigX3.value, self.P.value, self.T.value, self.Z.value

# if __name__ == "__main__":

#   RL_weights = False
  
#   if RL_weights:
#     W1_name = os.path.abspath(__file__ + "/../../../systems/nonlin_norm_weights/mlp_extractor.policy_net.0.weight.csv")
#     W2_name = os.path.abspath(__file__ + "/../../../systems/nonlin_norm_weights/mlp_extractor.policy_net.2.weight.csv")
#     W3_name = os.path.abspath(__file__ + "/../../../systems/nonlin_norm_weights/mlp_extractor.policy_net.4.weight.csv")
#     W4_name = os.path.abspath(__file__ + "/../../../systems/nonlin_norm_weights/action_net.weight.csv")
    
#     b1_name = os.path.abspath(__file__ + "/../../../systems/nonlin_norm_weights/mlp_extractor.policy_net.0.bias.csv")
#     b2_name = os.path.abspath(__file__ + "/../../../systems/nonlin_norm_weights/mlp_extractor.policy_net.2.bias.csv")
#     b3_name = os.path.abspath(__file__ + "/../../../systems/nonlin_norm_weights/mlp_extractor.policy_net.4.bias.csv")
#     b4_name = os.path.abspath(__file__ + "/../../../systems/nonlin_norm_weights/action_net.bias.csv")

#   else:
#     W1_name = os.path.abspath(__file__ + "/../../../../NN_training/int_nonlin/new_weights/mlp_extractor.policy_net.0.weight.csv")
#     W2_name = os.path.abspath(__file__ + "/../../../../NN_training/int_nonlin/new_weights/mlp_extractor.policy_net.2.weight.csv")
#     W3_name = os.path.abspath(__file__ + "/../../../../NN_training/int_nonlin/new_weights/mlp_extractor.policy_net.4.weight.csv")
#     W4_name = os.path.abspath(__file__ + "/../../../../NN_training/int_nonlin/new_weights/action_net.weight.csv")

#     b1_name = os.path.abspath(__file__ + "/../../../../NN_training/int_nonlin/new_weights/mlp_extractor.policy_net.0.bias.csv")
#     b2_name = os.path.abspath(__file__ + "/../../../../NN_training/int_nonlin/new_weights/mlp_extractor.policy_net.2.bias.csv")
#     b3_name = os.path.abspath(__file__ + "/../../../../NN_training/int_nonlin/new_weights/mlp_extractor.policy_net.4.bias.csv")
#     b4_name = os.path.abspath(__file__ + "/../../../../NN_training/int_nonlin/new_weights/action_net.bias.csv")

    
#   W1 = np.loadtxt(W1_name, delimiter=',')
#   W2 = np.loadtxt(W2_name, delimiter=',')
#   W3 = np.loadtxt(W3_name, delimiter=',')
#   W4 = np.loadtxt(W4_name, delimiter=',')
#   W4 = W4.reshape((1, len(W4)))

#   W = [W1, W2, W3, W4]

  
#   b1 = np.loadtxt(b1_name, delimiter=',')
#   b2 = np.loadtxt(b2_name, delimiter=',')
#   b3 = np.loadtxt(b3_name, delimiter=',')
#   b4 = np.loadtxt(b4_name, delimiter=',')
  
#   b = [b1, b2, b3, b4] 

#   lmi = LMI_final(W, b)
#   # alpha = lmi.search_alpha(1, 0, 1e-5, verbose=True)
#   # alpha = 0.0558
#   # lmi.solve(alpha, verbose=True)
#   # lmi.save_results('static_ETM')