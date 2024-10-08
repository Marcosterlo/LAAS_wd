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
constraints = [P >> 1e-3 * fake_zero(nx)]

T_val = cp.Variable(nphi)
T = cp.diag(T_val)
constraints += [T >> fake_zero(T.shape[0])]

Z1 = cp.Variable((neurons, nx))
Z2 = cp.Variable((neurons, nx))
Z3 = cp.Variable((neurons, nx))
Z4 = cp.Variable((neurons, nx))
Z = cp.vstack([Z1, Z2, Z3, Z4])

rho = cp.Variable()
constraints += [rho >> fake_zero(1)]

# Fixed parameters
alpha = 9*1e-4
P0 = np.array([[0.2916, 0.0054], [0.0054, 0.0090]])*1e5
vbar = 1

# Matrix creation
R = np.linalg.inv(np.eye(*Nvw.shape) - Nvw)
Rw = Nux + Nuw @ R @ Nvx
Rb = Nuw @ R @ Nvb + Nub
Abar = A + B @ K + B @ Nuw @ R @ Nvx

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
        vcap = np.min([np.abs(-vbar -s.wstar[i][k][0]), np.abs(vbar - s.wstar[i][k][0])], axis=0)
        ellip = cp.bmat([
            [P, cp.reshape(Z_el, (2,1))],
            [cp.reshape(Z_el, (1,2)), cp.reshape(2*alpha*T_el - alpha**2*vcap**(-2), (1, 1))] 
        ])
        constraints += [ellip >> 0]

Rphi = cp.bmat([
    [np.eye(nx), np.zeros((nx, nphi))],
    [R @ Nvx, np.eye(nphi) - R],
    [np.zeros((nphi, nx)), np.eye(nphi)]
])

decay_rate_eta_1 = 0.9
r1 = 0.8
l1 = decay_rate_eta_1 - r1
gamma1 = (l1 - 1) / r1
gamma1 = np.ones(neurons)*gamma1

decay_rate_eta_2 = 0.9
r2 = 0.8
l2 = decay_rate_eta_2 - r2
gamma2 = (l2 - 1) / r2
gamma2 = np.ones(neurons)*gamma2

decay_rate_eta_3 = 0.9
r3 = 0.8
l3 = decay_rate_eta_3 - r3
gamma3 = (l3 - 1) / r3
gamma3 = np.ones(neurons)*gamma3

decay_rate_eta_4 = 0.9
r4 = 0.8
l4 = decay_rate_eta_4 - r4
gamma4 = (l4 - 1) / r4
gamma4 = np.ones(neurons)*gamma4

gammavec = np.concatenate([gamma1, gamma2, gamma3, gamma4], axis=0)

gamma = cp.diag(gammavec)

mat = cp.bmat([
    [np.zeros((nx, nx)), np.zeros((nx, nphi)), np.zeros((nx, nphi))],
    [gamma @ Z, -gamma @ T, gamma @ T]
])

M = cp.vstack([Abar.T, (-B @ Nuw @ R).T]) @ P @ cp.hstack([Abar, -B @ Nuw @ R]) - cp.bmat([
    [P, np.zeros((nx, nphi))],
    [np.zeros((nphi, nx)), np.zeros((nphi, nphi))]
]) + Rphi.T @ mat.T + mat @ Rphi


constraints += [M << -fake_zero(M.shape[0])]
constraints += [M + rho * np.eye(M.shape[0]) >> fake_zero(M.shape[0])]

# Optimization condition
objective = cp.Minimize(rho)

# Problem definition
prob = cp.Problem(objective, constraints)

# Problem resolution
prob.solve(solver=cp.SCS, verbose=True, max_iter=100000)

# Checks
if prob.status not in  ["infeasible", "ubounded"]:
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