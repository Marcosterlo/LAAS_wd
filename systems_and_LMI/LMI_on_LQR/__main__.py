from systems_and_LMI.systems.LinearPendulum_integrator import LinPendulumIntegrator
import numpy as np
import cvxpy as cp

s = LinPendulumIntegrator()

K = np.load("K.npy")
A = s.A
B = s.B
nx = s.nx
Nux = np.zeros((1, 3))
Nuw = np.array([[1]])
Nub = np.array([[0]])
Nvx = K
Nvw = np.array([[0]])
Nvb = np.array([[0]])
R = np.linalg.inv(np.eye(Nvw.shape[0]) - Nvw)
Rw = Nux + Nuw @ R @ Nvx
Rb = Nuw @ R @ Nvb + Nub
Abar = A + B @ Rw

nphi = 1

P = cp.Variable((nx, nx), symmetric=True) 
T = cp.Variable((nphi, nphi))
Z = cp.Variable((nphi, nx))

Rphi = cp.bmat([
    [np.eye(nx), np.zeros((nx, nphi))],
    [R @ Nvx, np.eye(nphi) - R],
    [np.zeros((nphi, nx)), np.eye(nphi)]
])

mat = cp.bmat([
    [np.zeros((nx, nx)), np.zeros((nx, nphi)), np.zeros((nx, nphi))],
    [Z, -T, T]
])

M = cp.vstack([Abar.T, -(B @ Nuw @ R).T]) @ P @ cp.hstack([Abar, -(B @ Nuw @ R)]) - cp.bmat([
    [P, np.zeros((nx, nphi))],
    [np.zeros((nphi, nx)), np.zeros((nphi, nphi))]
]) - Rphi.T @ mat.T - mat @ Rphi

constraints = [P >> 0]
constraints += [T >= 0]
constraints += [M << -1e-3*np.eye(M.shape[0])]

vbar = 10.0
alpha = 9 * 1e-4

ellip = cp.bmat([
    [P, Z.T],
    [Z, cp.reshape(2*alpha*T - alpha**2*vbar**(-2), (1, 1))]
])
constraints += [ellip >> 0]

objective = cp.Minimize(cp.trace(P))

prob = cp.Problem(objective, constraints)

prob.solve(solver=cp.MOSEK, verbose=True)

if prob.status not in ["infeasible", "unbounded"]:
  print(f"Max eigenvalue of P: {np.max(np.linalg.eigvals(P.value))}")
  print(f"Max eigenvalue of M: {np.max(np.linalg.eigvals(M.value))}")