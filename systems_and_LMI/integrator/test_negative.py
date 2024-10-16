import numpy as np
from system import Integrator
import cvxpy as cp

s = Integrator()

M1 = np.load("M_mat.npy")
P = np.load("P_mat.npy")

# Values import
A = s.A
B = s.B
C = s.Mq.reshape(3, 1)
R = s.R
Rw = s.Rw
Nuw = s.N[1]

Abar = A + B @ Rw
Bbar = -B @ Nuw @ R

lam = cp.Variable((1, 1))

M2 = cp.bmat([
  [Abar.T @ P @ C],
  [Bbar.T @ P @ C]
])

M3 = cp.bmat([C.T @ P @ C])

to_check = M3 - M2.T @ np.linalg.inv(M1) @ M2

M = cp.bmat([
  [M1, M2],
  [M2.T, M3]
])

addition = cp.bmat([
  [np.zeros((1, 99)), -lam],
  [np.zeros((98, 100))],
  [-lam, np.zeros((1, 98)), -2*lam]
])

constraints = [lam >= 0]
constraints += [M + addition << -1e-4*np.eye(M.shape[0])]

prob = cp.Problem(cp.Maximize(lam), constraints)
prob.solve(solver=cp.MOSEK, verbose=True)

if prob.status == 'optimal':
  print(f"Lambda value: {lam.value[0][0]}")
  print(f"Max M eigenvalue: {np.max(np.linalg.eigvals((M + addition).value))}")
  print(f"Max P eigenvalue: {np.max(np.linalg.eigvals(P))}")