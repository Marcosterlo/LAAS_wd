from systems_and_LMI.systems.NonLinearPendulum_NN import NonLinPendulum_NN
import systems_and_LMI.systems.nonlin_dynamic_ETM.params as params
import numpy as np
import cvxpy as cp

# System initialization
s = NonLinPendulum_NN()

# Parameters unpacking
A = s.A
B = s.B
B = np.hstack([B, np.array([[0.], [0.], [1.]])])
C = s.C
nx = s.nx
nu = s.nu
nphi = s.nphi
vbar = s.bound
neurons = s.neurons
nlayer = s.nlayer
N = s.N
Nux = N[0]
Nux = np.vstack([Nux, np.zeros((1, Nux.shape[1]))])
Nuw = N[1]
Nuw = np.vstack([Nuw, np.zeros((1, Nuw.shape[1]))])
Nub = N[2]
Nub = np.vstack([Nub, np.zeros((1, Nub.shape[1]))])
Nvx = N[3]
Nvw = N[4]
Nvb = N[5]
R = s.R
Rw = Nux + Nuw @ R @ Nvx
Rb = Nuw @ R @ Nvb + Nub
nq = 1
gammavec = np.concatenate([params.gammas[i] * np.ones(neurons[i]) for i in range(nlayer - 1)])
gamma = cp.diag(gammavec)

# Auxiliary parameters
Abar = A + B @ Rw
Bbar = -B @ Nuw @ R

# Variables definition
P = cp.Variable((nx, nx), symmetric=True)

rho = cp.Variable((1, 1))

T_val = cp.Variable(nphi)
T = cp.diag(T_val)

Z1 = cp.Variable((neurons[0], nx))
Z2 = cp.Variable((neurons[1], nx))
Z3 = cp.Variable((neurons[2], nx))
Z = cp.vstack([Z1, Z2, Z3])

# Constraint matrices definition
Rphi = cp.bmat([
    [np.eye(nx), np.zeros((nx, nphi)), np.zeros((nx, nq))],
    [R @ Nvx, np.eye(R.shape[0]) - R, np.zeros((nphi, nq))],
    [np.zeros((nphi, nx)), np.eye(nphi), np.zeros((nphi, nq))],
    [np.zeros((nq, nx)), np.zeros((nq, nphi)), np.eye(nq)]
])

M1 = cp.bmat([
  [np.zeros((nx, nx)), np.zeros((nx, nphi)), np.zeros((nx, nphi)), np.zeros((nx, nq))],
  [gamma * Z, -gamma @ T , gamma @ T, np.zeros((nphi, nq))],
  [np.zeros((nq, nx)), np.zeros((nq, nphi)), np.zeros((nq, nphi)), np.zeros((nq, nq))]
])

Sinsec = cp.bmat([
  [0.0, -1.0],
  [-1.0, -2.0]
])

Rs = cp.bmat([
  [np.array([[1.0, 0.0, 0.0]]), np.zeros((1, nphi)), np.zeros((1, nq))],
  [np.zeros((nq, nx)), np.zeros((nq, nphi)), np.eye(nq)]
])

M = cp.bmat([
  [Abar.T @ P @ Abar - P, Abar.T @ P @ Bbar, Abar.T @ P @ C],
  [Bbar.T @ P @ Abar, Bbar.T @ P @ Bbar, Bbar.T @ P @ C],
  [C.T @ P @ Abar, C.T @ P @ Bbar, C.T @ P @ C]
]) - M1 @ Rphi - Rphi.T @ M1.T + Rs.T @ Sinsec @ Rs

# Constraints definition
constraints = [P >> 0]
constraints += [T >> 0]
constraints += [M << -1e-6 * np.eye(M.shape[0])]

# Ellipsoid conditions
for i in range(nlayer - 1):
  for k in range(neurons[i]):
    Z_el = Z[i*neurons[i] + k]
    T_el = T[i*neurons[i] + k, i*neurons[i] + k]
    vcap = np.min([np.abs(-vbar -s.wstar[i][k][0]), np.abs(vbar - s.wstar[i][k][0])], axis=0)
    ellip = cp.bmat([
        [P, cp.reshape(Z_el, (nx ,1))],
        [cp.reshape(Z_el, (1, nx)), cp.reshape(2*T_el - vcap**(-2), (1, 1))] 
    ])
    constraints += [ellip >> 0]

# Objective function definiton and problem solution
objective = cp.Minimize(cp.trace(P))

prob = cp.Problem(objective, constraints)

prob.solve(solver=cp.MOSEK, verbose=True)

if prob.status not in ['infeasible', 'unbounded']:
  print(f"Max P eigenvalue: {np.max(np.linalg.eigvals(P.value))}")
  print(f'Max M eigenvalue: {np.max(np.linalg.eigvals(M.value))}')