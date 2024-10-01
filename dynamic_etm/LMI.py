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
K = s.K

# NN parameters
nphi = s.nphi
W = s.W
b = s.b
nlayer = s.nlayer
nx = s.nx
nu = s.nu
neurons = int(nphi / (nlayer - 1))

# Creation of closed loop matrix N
N = block_diag(*W) # Reminder for myself: * operator unpacks lists and pass it as singular arguments

Nux = K
Nuw = N[nphi:, nx:]
Nub = b[-1]

Nvx = N[:nphi, :nx]
Nvw = N[:nphi, nx:]
Nvb = np.concatenate((b[0], b[1]))

# Fake zero function
def fake_zero(size):
    return 1e-3 * np.eye(size)

# rho and lambda creation
r = 0.42
l = 1 - r - 0.1
beta = (1 - l) / r

# Variables for LMI
P = cp.Variable((nx, nx), symmetric=True)
constraints = [P >> 1e-3 * fake_zero(nx)]

T_val = cp.Variable(nphi)
T = cp.diag(T_val)
constraints += [T >> fake_zero(T.shape[0])]

Z1 = cp.Variable((neurons, nx))
Z2 = cp.Variable((neurons, nx))
Z = cp.vstack([Z1, Z2])

rho = cp.Variable()
constraints += [rho >> fake_zero(1)]

# Fixed parameters
alpha = 9*1e-4
P0 = np.array([[0.2916, 0.0054], [0.0054, 0.0090]])
vbar = 1

# Matrix creation
R = np.linalg.inv(np.eye(*Nvw.shape) - Nvw)
Rw = Nux + Nuw @ R @ Nvx
Rb = Nuw @ R @ Nvb + Nub
Abar = A + B @ K + B @ Nuw @ R @ Nvx

# The fastes way I found to create big matrices was to create lines and stack them together
Rline1 = np.concatenate((np.eye(nx), np.zeros((2, 64))), axis=1)
Rline2 = np.concatenate((R @ Nvx, np.eye(nphi) - R), axis=1)
Rline3 = np.concatenate((np.zeros((nphi, nx)), np.eye(nphi)), axis=1)
Rphi = np.concatenate((Rline1, Rline2, Rline3), axis=0)

matline1 = np.concatenate((np.zeros((nx, nx)), np.zeros((nx, nphi)), np.zeros((nx, nphi))), axis=1)
matline2 = cp.hstack([Z, -T, T])
mat = cp.vstack([matline1, matline2])

M = cp.vstack([Abar.T, (-B @ Nuw @ R).T]) @ P @ cp.hstack([Abar, -B @ Nuw @ R]) - cp.bmat([
    [P, np.zeros((nx, nphi))],
    [np.zeros((nphi, nx)), np.zeros((nphi, nphi))]
]) + beta * (- Rphi.T @ mat.T - mat @ Rphi)
 
# Definite negative M constraint for stability
constraints += [M << -fake_zero(M.shape[0])]

# Condition to reach thhe lmimits of feasibility
constraints += [M + rho * np.eye(M.shape[0]) >> fake_zero(M.shape[0])]

# Inclusion constraint
inclusion = cp.bmat([
    [cp.hstack([P0, P])],
    [cp.hstack([P, P])]
])
constraints += [inclusion >> fake_zero(inclusion.shape[0])]

# Ellipsoid condition
for i in range(nlayer-1):
    for k in range(neurons):
        Z_el = Z[i*neurons + k]
        T_el = T[i*neurons + k, i*neurons + k]
        ellip = cp.bmat([
            [P, cp.reshape(Z_el, (2,1))],
            [cp.reshape(Z_el, (1,2)), cp.reshape(2*alpha*T_el - alpha**2*vbar**(-2), (1, 1))] 
        ])
        constraints += [ellip >> 0]

# Optimization condition
objective = cp.Minimize(rho)

# Problem definition
prob = cp.Problem(objective, constraints)

# Problem resolution
prob.solve(solver=cp.SCS, verbose=True)

# Checks
if prob.status not in  ["infeasible", "ubounded"]:
    print("Problem status is " + prob.status)
    print("Max P eigenvalue: ", np.max(np.linalg.eigvals(P.value)))
    print("Max M eigenvalue: ", np.max(np.linalg.eigvals(M.value)))
    print("Max T eigenvalue: ", np.max(np.linalg.eigvals(T.value)))

    # Saving matrices to npy file
    np.save("P_mat", P.value)
    np.save("Z_mat", Z.value)
    np.save("T_mat", T.value)
else:
    print("=========== Unfeasible problem =============")