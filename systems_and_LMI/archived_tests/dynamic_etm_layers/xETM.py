import cvxpy as cp
from system import System
import numpy as np
from scipy.linalg import block_diag

s = System()

A = s.A
B = s.B
K = s.K

nphi = s.nphi
W = s.W
b = s.b
nlayer = s.nlayer
nx = s.nx
nu = s.nu
neurons = int(nphi / (nlayer - 1))

N = block_diag(*W)

Nux = K
Nuw = N[nphi:, nx:]
Nub = b[-1]

Nvx = N[:nphi, :nx]
Nvw = N[:nphi, nx:]
Nvb = np.concatenate((b[0], b[1]))

P = cp.Variable((nx, nx), symmetric=True)
constraints = [P >> 0]

T_val = cp.Variable(nphi)
T = cp.diag(T_val)
T1 = T[:neurons, :neurons]
T2 = T[neurons:, neurons:]
constraints += [T >> 0]

Z1 = cp.Variable((neurons, nx))
Z2 = cp.Variable((neurons, nx))
Z = cp.vstack([Z1, Z2])

alpha = cp.Parameter(nonneg=True)
vbar = 1

R = np.linalg.inv(np.eye(*Nvw.shape) - Nvw)
Rw = Nux + Nuw @ R @ Nvx
Rb = Nuw @ R @ Nvb + Nub
Abar = A + B @ K + B @ Nuw @ R @ Nvx
Bbar = -B @ Nuw @ R

for i in range (nlayer - 1):
  for k in range(neurons):
    Z_el = Z[i*neurons + k]
    T_el = T[i*neurons + k, i*neurons + k]
    vcap = np.min([np.abs(-vbar - s.wstar[i][k][0]), np.abs(vbar - s.wstar[i][k][0])], axis=0)
    ellip = cp.bmat([
      [P, cp.reshape(Z_el, (2, 1))],
      [cp.reshape(Z_el, (1, 2)), cp.reshape(2*alpha*T_el - alpha**2*vcap**(-2), (1, 1))]
    ])
    constraints += [ellip >> 0]

Rphi = cp.bmat([
  [np.eye(nx), np.zeros((nx, nphi))],
  [R @ Nvx, np.eye(nphi) - R],
  [np.zeros((nphi, nx)), np.eye(nphi)]
])

decay_rate_eta_1 = 0.9
r1 = 0.6
l1 = decay_rate_eta_1 - r1
gamma1 = (l1 - 1) / r1
gamma1 = np.ones(nphi + nx) * gamma1

decay_rate_eta_2 = 0.9
r2 = 0.4
l2 = decay_rate_eta_2 - r2
gamma2 = (l2 - 1) / r2
gamma2 = np.ones(nphi + nx) * gamma2

gammavec = np.concatenate([gamma1, gamma2], axis=0)

gamma = cp.diag(gammavec)

# New variables
nbigx1 = neurons * 2 + nx
bigX1 = cp.Variable((nbigx1, nbigx1))
nbigx2 = neurons * 2 + nx
bigX2 = cp.Variable((nbigx2, nbigx2))

bigX = cp.bmat([
  [bigX1, np.zeros((nbigx1, nbigx2))],
  [np.zeros((nbigx2, nbigx1)), bigX2]
])

Omega1 = cp.bmat([
  [np.zeros((nx, nx)), np.zeros((nx, neurons)), np.zeros((nx, neurons))],
  [Z1, T1, -T1],
  [np.zeros((neurons, nx)), np.zeros((neurons, neurons)), np.zeros((neurons, neurons))]
])

Omega2 = cp.bmat([
  [np.zeros((nx, nx)), np.zeros((nx, neurons)), np.zeros((nx, neurons))],
  [Z2, T2, -T2],
  [np.zeros((neurons, nx)), np.zeros((neurons, neurons)), np.zeros((neurons, neurons))]
])

Omega = cp.bmat([
  [Omega1, np.zeros((Omega1.shape[0], Omega2.shape[1]))],
  [np.zeros((Omega2.shape[0], Omega1.shape[1])), Omega2]
])

idx = np.eye(nx) * 1 / (nlayer - 1)
xzero = np.zeros((nx, neurons))

id = np.eye(neurons)
zerox = np.zeros((neurons, nx))
zero = np.zeros((neurons, neurons))

Rxi = cp.bmat([
  [idx, xzero, xzero, xzero, xzero],
  [zerox, id, zero, zero, zero],
  [zerox, zero, zero, id, zero],
  [idx, xzero, xzero, xzero, xzero],
  [zerox, zero, id, zero, zero],
  [zerox, zero, zero, zero, id]
])

Rnu = cp.bmat([
  [np.eye(nx), np.zeros((nx, nphi))],  
  [np.zeros((nphi, nx)), np.eye(nphi)],
  [R @ Nvx, np.eye(R.shape[0]) - R]
])
  
hconstr = cp.hstack([R @ Nvx, np.eye(R.shape[0]) - R, -np.eye(nphi)])

N1 = cp.Variable((nx, nphi))
N2 = cp.Variable((nphi, nphi))
N3 = cp.Variable((nphi, nphi))
N = cp.vstack([N1, N2, N3])

newconstr = Rxi.T @ (bigX - Omega + bigX.T - Omega.T) @ Rxi + N @ hconstr + hconstr.T @ N.T

constraints += [newconstr << 0]

M = cp.bmat([
  [Abar.T @ P @ Abar - P, Abar.T @ P @ Bbar],
  [Bbar.T @ P @ Abar, Bbar.T @ P @ Bbar]
]) + Rnu.T @ Rxi.T @ (gamma @ bigX + bigX.T @ gamma.T) @ Rxi @ Rnu

constraints += [M << -1e-6 * np.eye(M.shape[0])]
objective = cp.Minimize(cp.trace(P))

prob = cp.Problem(objective, constraints)

alpha.value = 9 * 1e-4

prob.solve(solver=cp.MOSEK, verbose=True)
print(f"Max eigenvalue of P: {np.max(np.linalg.eigvals(P.value))}")
print(f"Max eigenvalue of M: {np.max(np.linalg.eigvals(M.value))}")

np.save('mat-weights/bigX1.npy', bigX1.value)
np.save('mat-weights/bigX2.npy', bigX2.value)
np.save('mat-weights/xP.npy', P.value)
np.save('mat-weights/xT.npy', T.value)
np.save('mat-weights/xZ.npy', Z.value)


# feasible_extreme = 0.1
# infeasible_extreme = 0.0
# threshold = 1e-6
# golden_ratio = (1 + np.sqrt(5)) / 2
# error = False

# try:
#   while (feasible_extreme - infeasible_extreme > threshold):
#     x1 = feasible_extreme - (feasible_extreme - infeasible_extreme) / golden_ratio
#     x2 = infeasible_extreme + (feasible_extreme - infeasible_extreme) / golden_ratio
#     alpha.value = x1
#     prob.solve(solver=cp.MOSEK, verbose=True)
#     if prob.status in ["infeasible", "unbounded", "unknown"]:
#       fx1 = 1e5
#     else:
#       fx1 = np.max(np.linalg.eigvals(P.value))
#       print(f"\n==================== \nMax eigenvalue of P: {fx1}")
#       print(f"Max eigenvalue of M: {np.max(np.linalg.eigvals(M.value))}")
#       print(f"Current alpha value: {alpha.value}\n==================== \n")
#     alpha.value = x2
#     prob.solve(solver=cp.MOSEK, verbose=True)
#     if prob.status in ["infeasible", "unbounded", "unknown"]:
#       fx2 = 1e5
#     else:
#       fx2 = np.max(np.linalg.eigvals(P.value))
#       print(f"\n==================== \nMax eigenvalue of P: {fx2}")
#       print(f"Max eigenvalue of M: {np.max(np.linalg.eigvals(M.value))}")
#       print(f"Current alpha value: {alpha.value}\n==================== \n")
#     if fx1 < fx2:
#       feasible_extreme = x2
#     else:
#       infeasible_extreme = x1
# except cp.error.SolverError:
#   error = True
#   print("Solver error")
#   print(f"Max eigenvalue of P: {np.max(np.linalg.eigvals(P.value))}")
#   print(f"Max eigenvalue of M: {np.max(np.linalg.eigvals(M.value))}")
#   print(f"Final alpha value: {alpha.value}")

# if not error:
#   print(f"Max eigenvalue of P: {np.max(np.linalg.eigvals(P.value))}")
#   print(f"Max eigenvalue of M: {np.max(np.linalg.eigvals(M.value))}")
#   print(f"Final alpha value: {alpha.value}")