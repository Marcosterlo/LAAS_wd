import cvxpy as cp
from LinearPendulum import LinearPendulum
import numpy as np

s = LinearPendulum()

A = s.A
B = s.B
R = s.R
Rw = s.Rw
Rb = s.Rb
N = s.N
Nux = N[0]
Nuw = N[1]
Nub = N[2]
Nvx = N[3]
Nvw = N[4]
Nvb = N[5]
nphi = s.nphi
nx = s.nx
nu = s.nu
neurons = [32, 32, 32]
nlayer = s.nlayer

P = cp.Variable((nx, nx), symmetric=True)
transf = cp.bmat([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
])
P_ellip = transf @ P @ transf
T_val = cp.Variable(nphi)
T = cp.diag(T_val)
Z1 = cp.Variable((neurons[0], nx))
Z2 = cp.Variable((neurons[1], nx))
Z3 = cp.Variable((neurons[2], nx))
Z = cp.vstack([Z1, Z2, Z3])

Abar = A + B @ Rw

Rphi = cp.bmat([
    [np.eye(nx), np.zeros((nx, nphi))],
    [R @ Nvx, np.eye(nphi) - R],
    [np.zeros((nphi, nx)), np.eye(nphi)]
])

d1 = 0.9
r1 = 0.4
l1 = d1 - 41
g1 = (1 - l1) / r1
g1 = np.ones(neurons[0]) * g1

d2 = 0.9
r2 = 0.3
l2 = d2 - r2
g2 = (1 - l2) / r2
g2 = np.ones(neurons[1]) * g2

d3 = 0.9
r3 = 0.2
l3 = d3 - r3
g3 = (1 - l3) / r3
g3 = np.ones(neurons[2]) * g3

gammav = np.concatenate([g1, g2, g3], axis=0)
gamma = cp.diag(gammav)

mat = cp.bmat([
    [np.zeros((nx, nx)), np.zeros((nx, nphi)), np.zeros((nx, nphi))],
    [gamma @ Z, -gamma @ T, gamma @ T]
])

M = cp.vstack([Abar.T, (-B @ Nuw @ R).T]) @ P @ cp.hstack([Abar, -B @ Nuw @ R]) - cp.bmat([
    [P, np.zeros((nx, nphi))],
    [np.zeros((nphi, nx)), np.zeros((nphi, nphi))]
]) - Rphi.T @ mat.T - mat @ Rphi

constraints = [P >> 0]
constraints += [T >> 0]
constraints += [M << -1e-3*np.eye(M.shape[0])]

vbar = 1
alpha = 9 * 1e-4

for i in range(nlayer - 1):
  for k in range(len(neurons)):
    Z_el = Z[i*neurons[i] + k]
    T_el = T[i*neurons[i] + k, i*neurons[i] + k]
    vcap = np.min([np.abs(-vbar - s.wstar[i][k][0]), np.abs(vbar - s.wstar[i][k][0])], axis=0)
    ellip = cp.bmat([
        [P_ellip, cp.reshape(Z_el, (3,1))],
        [cp.reshape(Z_el, (1, 3)), cp.reshape(2*alpha*T_el - alpha**2*vcap**(-2), (1, 1))] 
    ])
    constraints += [ellip >> 0]

objective = cp.Minimize(cp.trace(P))

prob = cp.Problem(objective, constraints)

prob.solve(solver=cp.MOSEK, verbose=True)

if prob.status not in ["infeasible", "unbounded"]:
  print(f"Max P eigenvalue: {np.max(np.linalg.eigvals(P.value))}")
  print(f"Max M eigenvalue: {np.max(np.linalg.eigvals(M.value))}")
  np.save('./dynamic_ETM/P.npy', P.value)
  np.save('./dynamic_ETM/T.npy', T.value)
  np.save('./dynamic_ETM/Z.npy', Z.value)