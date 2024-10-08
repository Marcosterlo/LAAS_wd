import cvxpy as cp
from system import System
import numpy as np

# System initialization
s = System()

# Matrix import
A = s.A
B = s.B
K = s.K

# NN parameters
nphi = s.nphi
nlayer = s.nlayer
nx = s.nx
nu = s.nu
neurons = int(nphi / (nlayer - 1))

N = s.N
Nux = N[0]
Nuw = N[1]
Nub = N[2]
Nvx = N[3]
Nvw = N[4]
Nvb = N[5]

# Fake zero function
def fake_zero(size):
    return 1e-3 * np.eye(size)

# Variables for LMI
P = cp.Variable((nx, nx), symmetric=True)
constraints = [P >> fake_zero(nx)]

T_val = cp.Variable(nphi)
T = cp.diag(T_val)
constraints += [T >> fake_zero(T.shape[0])]

Z1 = cp.Variable((neurons, nx))
Z2 = cp.Variable((neurons, nx))
Z3 = cp.Variable((neurons, nx))
Z4 = cp.Variable((neurons, nx))
Z = cp.vstack([Z1, Z2, Z3, Z4])

# Matrix creation
R = np.linalg.inv(np.eye(*Nvw.shape) - Nvw)
Rw = Nux + Nuw @ R @ Nvx
Rb = Nuw @ R @ Nvb + Nub
Abar = A + B @ K + B @ Nuw @ R @ Nvx

Rphi = cp.bmat([
    [np.eye(nx), np.zeros((nx, nphi))],
    [R @ Nvx, np.eye(nphi) - R],
    [np.zeros((nphi, nx)), np.eye(nphi)]
])

mat = cp.bmat([
    [np.zeros((nx, nx)), np.zeros((nx, nphi)), np.zeros((nx, nphi))],
    [Z, -T, T]
])

M = cp.vstack([Abar.T, (-B @ Nuw @ R).T]) @ P @ cp.hstack([Abar, -B @ Nuw @ R]) - cp.bmat([
    [P, np.zeros((nx, nphi))],
    [np.zeros((nphi, nx)), np.zeros((nphi, nphi))]
]) - Rphi.T @ mat.T - mat @ Rphi


constraints += [M << -fake_zero(M.shape[0])]

# Optimization condition
objective = cp.Minimize(cp.trace(P))

# Problem definition
prob = cp.Problem(objective, constraints)

# Problem resolution
prob.solve(solver=cp.SCS, verbose=True, max_iters=1000000)

# Checks
if prob.status not in  ["infeasible", "ubounded", "unbounded_inaccurate"]:
    print("Problem status is " + prob.status)
    print("Max P eigenvalue: ", np.max(np.linalg.eigvals(P.value)))
    print("Max M eigenvalue: ", np.max(np.linalg.eigvals(M.value)))
    print("Max T eigenvalue: ", np.max(np.linalg.eigvals(T.value)))

    # Saving matrices to npy file
    # np.save("P_mat", P.value)
    # np.save("Z_mat", Z.value)
    # np.save("T_mat", T.value)
else:
    print("=========== Unfeasible problem =============")