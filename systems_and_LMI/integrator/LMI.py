import cvxpy as cp
import numpy as np
from system import Integrator

s = Integrator()

# Values import
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
g = s.g
l = s.l
dt = s.dt
nlayer = s.nlayer

# Variables
Ptrue = cp.Variable((nx-1, nx-1), symmetric=True)
P = cp.bmat([
    [cp.hstack([Ptrue, np.zeros((2, 1))])],
    [np.array([0, 0, 0]).reshape(1,3)]
])
T_val = cp.Variable(nphi)
T = cp.diag(T_val)
Z1 = cp.Variable((neurons[0], nx))
Z2 = cp.Variable((neurons[1], nx))
Z3 = cp.Variable((neurons[2], nx))
Z = cp.vstack([Z1, Z2, Z3])*0

# Composite matrices
Abar = A + B @ Rw

mat1 = cp.bmat([
  [np.zeros((nx, nx)), np.zeros((nx, nphi)), np.zeros((nx, nphi))],
  [Z, -T, T],
  [np.zeros((1, nx)), np.zeros((1, nphi)), np.zeros((1, nphi))]
])

Rphi = cp.bmat([
  [np.eye(nx), np.zeros((nx, nphi)), np.zeros((nx, 1))],
  [R @ Nvx, np.eye(R.shape[0]) - R, np.zeros((nphi, 1))],
  [np.zeros((nphi, nx)), np.eye(nphi), np.zeros((nphi, 1))]
])

Rq = cp.bmat([
  [np.hstack([1, 0, 0, np.zeros(nphi), 0]).reshape(1, nx + nphi + 1)],
  [np.hstack([0, 0, 0, np.zeros(nphi), 1]).reshape(1, nx + nphi + 1)]
])

sec_mat = cp.bmat([
  [0, -1],
  [-1, -2]
])

P_mat = cp.bmat([
  [P, np.zeros((nx, nphi)), np.zeros((nx, 1))],
  [np.zeros((nphi, nx)), np.zeros((nphi, nphi)), np.zeros((nphi, 1))],
  [np.hstack([np.zeros(nx), np.zeros(nphi), 0]).reshape(1, nx + nphi + 1)]
])

q = np.array([[0],
             [g/l*dt],
             [0]]) 

M = cp.vstack([Abar.T, (-B @ Nuw @ R).T, q.T]) @ P @ cp.hstack([Abar, -B @ Nuw @ R, q]) - P_mat - Rphi.T @ mat1.T - mat1 @ Rphi - Rq.T @ sec_mat @ Rq

## Constraints
constraints = [P >> 0]
constraints += [T >> 0]
constraints += [M << -1e-5*np.eye(M.shape[0])]

vbar = 1
alpha = cp.Variable()
constraints += [alpha >= 0]

# Ellipsoid condition
for i in range(nlayer-1):
    for k in range(neurons[i]):
        Z_el = Z[i*neurons[i] + k]
        T_el = T[i*neurons[i] + k, i*neurons[i] + k]
        vcap = np.min([np.abs(-vbar -s.wstar[i][k][0]), np.abs(vbar - s.wstar[i][k][0])], axis=0)
        ellip = cp.bmat([
            [P, cp.reshape(Z_el, (3,1))],
            [cp.reshape(Z_el, (1,3)), cp.reshape(2*alpha*T_el - alpha**2*vcap**(-2), (1, 1))] 
        ])
        constraints += [ellip >> 0]

objective = cp.Minimize(alpha)
prob = cp.Problem(objective, constraints)

# Problem resolution
solved = False
prob.solve(solver=cp.MOSEK, verbose=True)
if prob.status not in  ["infeasible", "ubounded", "unbounded_inaccurate", "infeasible_inaccurate"]:
    solved = True
else:
    print("Mosek failed")

if not solved:
    prob.solve(solver=cp.SCS, verbose=True)

if prob.status not in  ["infeasible", "ubounded", "unbounded_inaccurate", "infeasible_inaccurate"]:
    solved = True

if solved:
    print("Problem status is " + prob.status)
    print("Max P eigenvalue: ", np.max(np.linalg.eigvals(P.value)))
    print("Max M eigenvalue: ", np.max(np.linalg.eigvals(M.value)))
    print("Max T eigenvalue: ", np.max(np.linalg.eigvals(T.value)))
else:
    print("=========== Unfeasible problem =============")