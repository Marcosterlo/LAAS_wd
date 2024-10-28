from systems_and_LMI.systems.NonLinearPendulum_NN import NonLinPendulum_NN
import systems_and_LMI.systems.nonlin_dynamic_ETM.params as params
import numpy as np
import cvxpy as cp

# System initialization
s = NonLinPendulum_NN()

# Parameters unpacking
A = s.A
B = s.B
C = s.C
D = s.D
nx = s.nx
nu = s.nu
nphi = s.nphi
vbar = s.bound
neurons = s.neurons
nlayer = s.nlayer
N = s.N
Nux = N[0]
Nuw = N[1]
Nub = N[2]
Nvx = N[3]
Nvw = N[4]
Nvb = N[5]
R = s.R
Rw = s.Rw
Rb = s.Rb
nq = 1
nr = 1
gammavec = np.concatenate([params.gammas[i] * np.ones(neurons[i]) for i in range(nlayer - 1)])
gamma = cp.diag(gammavec)

# Auxiliary parameters
Abar = A + B @ Rw
Bbar = -B @ Nuw @ R
Cx = np.zeros((nphi, nx))
Cx[:, 0] = 1.0
Cr = np.ones((nphi, nr))

# Variables definition
P = cp.Variable((nx, nx), symmetric=True)

T_val = cp.Variable(nphi)
T = cp.diag(T_val)

Z1 = cp.Variable((neurons[0], nx))
Z2 = cp.Variable((neurons[1], nx))
Z3 = cp.Variable((neurons[2], nx))
Z = cp.vstack([Z1, Z2, Z3])

Omega_val = cp.Variable(nphi)
Omega = cp.diag(Omega_val)

# Constraint matrices definition
Mpi = cp.bmat([
  [-Cx.T @ gamma @ Omega @ Cx, np.zeros((nx, nphi)), np.zeros((nx, nphi)), np.zeros((nx, nq)), Cx.T @ gamma @ Omega @ Cr],
  [gamma @ Z, -gamma @ T, gamma @ T, np.zeros((nphi, nq)), np.zeros((nphi, nr))],
  [np.zeros((nq, nx)), np.zeros((nq, nphi)), np.zeros((nq, nphi)), np.zeros((nq, nq)), np.zeros((nq, nr))],
  [Cr.T @ gamma @ Omega @ Cx, np.zeros((nr, nphi)), np.zeros((nr, nphi)), np.zeros((nr, nq)), -Cr.T @ gamma @ Omega @ Cr]
])

Rpi = cp.bmat([
  [np.eye(nx), np.zeros((nx, nphi)), np.zeros((nx, nq)), np.zeros((nx, nr))],
  [R @ Nvx, np.eye(R.shape[0]) - R, np.zeros((nphi, nq)), np.zeros((nphi, nr))],
  [np.zeros((nphi, nx)), np.eye(nphi), np.zeros((nphi, nq)), np.zeros((nphi, nr))],
  [np.zeros((nq, nx)), np.zeros((nq, nphi)), np.eye(nq), np.zeros((nq, nr))],
  [np.zeros((nr, nx)), np.zeros((nr, nphi)), np.zeros((nr, nq)), np.eye(nr)]
])

Rs = cp.bmat([
  [np.array([[1.0, 0.0, 0.0]]), np.zeros((1, nphi + nq + nr))],
  [np.zeros((1, nx)), np.zeros((1, nphi)), np.eye(nq), np.zeros((1, nr))]
])

Sinsec = np.array([[0.0, -1.0],
                   [-1.0, -2.0]])

M = cp.bmat([
  [Abar.T @ P @ Abar - P, Abar.T @ P @ Bbar, Abar.T @ P @ C, np.zeros((nx, nr))],
  [Bbar.T @ P @ Abar, Bbar.T @ P @ Bbar, Bbar.T @ P @ C, np.zeros((nphi, nr))],
  [C.T @ P @ Abar, C.T @ P @ Bbar, C.T @ P @ C, np.zeros((nq, nr))],
  [np.zeros((nr, nx)), np.zeros((nr, nphi)), np.zeros((nr, nq)), np.zeros((nr, nr))]
]) - Mpi @ Rpi - Rpi.T @ Mpi.T + Rs.T @ Sinsec @ Rs

# Constraints definiton