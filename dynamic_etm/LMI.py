import cvxpy as cp
from system import System
import numpy as np
from scipy.linalg import block_diag

# System initialization
s = System()

# Empty list of constraints
pbLMI = []

# Matrix import
A = s.A
B = s.B

# NN parameters
nphi = s.nphi
W = s.W
b = s.b
nlayer = s.nlayer
nx = s.nx
nu = s.nu

# Creation of closed loop matrix N
N = block_diag(*W) # Reminder for myself: * operator unpacks lists and pass it as singular arguments

Nux = N[nphi:, :nx]
Nuw = N[nphi:, nx:]
Nub = b[-1]

Nvx = N[:nphi, :nx]
Nvw = N[:nphi, nx:]
Nvb = np.concatenate((b[0], b[1]))

# Variables for LMI
P = cp.Variable((nx, nx), symmetric=True)
constraints = [P >> 1e-3 * np.eye(nx)]

rho = 0.2
lam = 0.2
tau = 1e3

psi1 = cp.Variable((nphi, nphi))
psi2 = cp.Variable((nphi, nphi))
psi3 = cp.Variable((nphi, nphi))
constraints += [psi1 + psi2 + psi2.T + psi3 << 0]

S = cp.Variable((nphi, nphi), symmetric=True)
R = cp.Variable((nphi, nphi), symmetric=True)
T = cp.Variable((nphi, nphi))
constraints += [S >> 0]
constraints += [R << 0]

# Matrix M creation

# Non null terms
lmi11 = A.T @ P @ A - P + A.T @ P @ B @ Nux + Nux.T @ B.T @ P @ A + Nux.T @ B.T @ P @ B @ Nux + tau * Nvx.T @ S @ Nvx
lmi13 = A.T @ P @ B @ Nuw + Nux.T @ B.T @ P @ B @ Nuw + tau * (Nvx.T @ S @ Nvw + Nvx.T @ T)
lmi22 = (lam - 1)/rho * psi3
lmi23 = (lam - 1)/rho * psi2.T
lmi33 = Nuw.T @ B.T @ P @ B @ Nuw + (lam - 1)/rho * psi1 + tau * (R + Nvw.T @ S @ Nvw + Nvw.T @ T + T.T @ Nvw)

# Null terms
lmi12 = np.zeros((lmi11.shape[0], lmi22.shape[1]))

M = cp.bmat([
    [lmi11, lmi12, lmi13],
    [lmi12.T, lmi22, lmi23],
    [lmi13.T, lmi23.T, lmi33]
])

constraints += [M << -1e-3]

objective = cp.Minimize(cp.trace(P))

prob = cp.Problem(objective, constraints)

prob.solve(solver=cp.MOSEK, verbose=True)