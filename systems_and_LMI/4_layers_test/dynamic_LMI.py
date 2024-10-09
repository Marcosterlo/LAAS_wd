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

dec = 0.9
r1 = 0.8
r2 = 0.7
r3 = 0.6
r4 = 0.5
lam1 = dec - r1
lam2 = dec - r2
lam3 = dec - r3
lam4 = dec - r4
gamma1 = (1 - lam1) / r1
gamma2 = (1 - lam2) / r2
gamma3 = (1 - lam3) / r3
gamma4 = (1 - lam4) / r4

gammavec = np.concatenate([gamma1, gamma2, gamma3, gamma4], axis=0)
gamma = cp.diag(gammavec)

mat = cp.bmat([
    [np.zeros((nx, nx)), np.zeros((nx, nphi)), np.zeros((nx, nphi))],
    [gamma @ Z, -gamma @ T, gamma @ T]
])


M = cp.vstack([Abar.T, (-B @ Nuw @ R).T]) @ P @ cp.hstack([Abar, -B @ Nuw @ R]) - cp.bmat([
    [P, np.zeros((nx, nphi))],
    [np.zeros((nphi, nx)), np.zeros((nphi, nphi))]
]) - Rphi.T @ mat.T - mat @ Rphi


constraints += [M << -fake_zero(M.shape[0])]

# Inclusion constraint
P0 = np.load('./4_layers/P0_mat.npy')
inclusion = cp.bmat([
    [cp.hstack([P0, P])],
    [cp.hstack([P, P])]
])
constraints += [inclusion >> fake_zero(inclusion.shape[0])]

rho = cp.Variable()
constraints += [rho >> fake_zero(1)]

alpha = 9*1e-4
vbar = 1

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

constraints += [M + rho * np.eye(M.shape[0]) >> fake_zero(M.shape[0])]

# Optimization condition
objective = cp.Minimize(rho)

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