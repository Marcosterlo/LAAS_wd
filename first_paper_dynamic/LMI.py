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
neurons = int(nphi / (nlayer - 1))

# Creation of closed loop matrix N
N = block_diag(*W) # Reminder for myself: * operator unpacks lists and pass it as singular arguments

Nux = N[nphi:, :nx]
Nuw = N[nphi:, nx:]
Nub = b[-1]

Nvx = N[:nphi, :nx]
Nvw = N[:nphi, nx:]
Nvb = np.concatenate((b[0], b[1]))

# Fake zero function
def fake_zero(size):
    return 1e-3 * np.eye(size)

# Variables for LMI
P = cp.Variable((nx, nx), symmetric=True)
constraints = [P >> 1e-3 * fake_zero(nx)]

T_val = cp.Variable(nphi)
T = cp.diag(T_val)
constraints += [T >> fake_zero(T.shape[0])]

rho = cp.Variable()
constraints += [rho >> fake_zero(1)]

decay_rate = 0.9
r = 0.1
l = 0.1
beta = (1 - l)/r

qdelta1 = cp.Variable((neurons, neurons))
qdelta2 = cp.Variable((neurons, neurons))
Qdelta = cp.bmat([
    [cp.hstack([qdelta1, np.zeros((neurons, neurons))])],
    [cp.hstack([np.zeros((neurons, neurons)), qdelta2])]
])
constraints += [Qdelta << np.eye(Qdelta.shape[0])]

qw1 = cp.Variable((neurons, neurons))
qw2 = cp.Variable((neurons, neurons))
Qw = cp.bmat([
    [cp.hstack([qw1, np.zeros((neurons, neurons))])],
    [cp.hstack([np.zeros((neurons, neurons)), qw2])]
])
constraints += [Qw >> rho * np.eye(Qw.shape[0])]

Z1 = cp.Variable((neurons, nx))
Z2 = cp.Variable((neurons, nx))
Z = cp.vstack([Z1, Z2])

# Fixed parameters
alpha = 9*1e-4
P0 = np.array([[0.3024, 0.0122], [0.0122, 0.0154]])
vbar = 1

# Matrix creation
R = np.linalg.inv(np.eye(*Nvw.shape) - Nvw)
Rw = Nux + Nuw @ R @ Nvx
Rb = Nuw @ R @ Nvb + Nub

# The fastes way I found to create big matrices was to create lines and stack them together
Rphi = cp.bmat([
    [cp.hstack([np.eye(nx), np.zeros((nx, nphi)), np.zeros((nx, nphi))])],
    [cp.hstack([Nvx, Nvw, Nvw])],
    [cp.hstack([np.zeros((nphi, nx)), np.eye(nphi), np.zeros((nphi, nphi))])]
])

mat = cp.bmat([
    [cp.hstack([np.zeros((nx, nx)), -Z.T, Z.T])],
    [cp.hstack([-Z, np.zeros((nphi, nphi)), T])],
    [cp.hstack([Z, T.T, -2*T])]
])

M = cp.bmat([
    [cp.hstack([(A + B @ Nux).T @ P @ (A + B @ Nux) - P, (A + B @ Nux).T @ P @ B @ Nuw, (A + B @ Nux).T @ P @ B @ Nuw])],
    [cp.hstack([((A + B @ Nux).T @ P @ B @ Nuw).T, (B @ Nuw).T @ P @ B @ Nuw + Qw, (B @ Nuw).T @ P @ B @ Nuw])],
    [cp.hstack([((A + B @ Nux).T @ P @ B @ Nuw).T, ((B @ Nuw).T @ P @ B @ Nuw).T, (B @ Nuw).T @ P @ B @ Nuw - Qdelta])]
    ]) + beta * Rphi.T @ mat @ Rphi
 
# Definite negative M constraint for stability
constraints += [M << -fake_zero(M.shape[0])]

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

# Optimization condition
objective = cp.Minimize(-rho)

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
    # np.save("P_mat", P.value)
else:
    print("=========== Unfeasible problem =============")