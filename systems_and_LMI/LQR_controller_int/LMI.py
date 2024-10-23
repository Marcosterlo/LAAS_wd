from systems_and_LMI.systems.LinearPendulum_integrator import LinPendulumIntegrator
import numpy as np
import cvxpy as cp

# Systme initialization
s = LinPendulumIntegrator()

# Unpacking of the system parameters
K = np.load("K.npy")
A = s.A
B = s.B
nx = s.nx
nphi = 1
vbar = s.max_torque
alpha = 1
Ak = A - B @ K

# Variables definition
P = cp.Variable((nx, nx), symmetric=True) 
T = cp.Variable((nphi, nphi))
Z = cp.Variable((nphi, nx))

# Constraint matrices definition
Rphi = cp.bmat([
    [np.eye(nx), np.zeros((nx, nphi))],
    [-K, np.array([[0.0]])],
    [np.zeros((nphi, nx)), np.eye(nphi)]
])

mat = cp.bmat([
    [np.zeros((nx, nx)), np.zeros((nx, nphi)), np.zeros((nx, nphi))],
    [Z, -T, T]
])


M = cp.bmat([
    [Ak.T @ P @ Ak - P, -Ak.T @ P @ B],
    [-B.T @ P @ Ak, B.T @ P @ B]
]) - Rphi.T @ mat.T - mat @ Rphi

ellip = cp.bmat([
    [P, Z.T],
    [Z, cp.reshape(2*alpha*T - alpha**2*vbar**(-2), (1, 1))]
])

# Constraints definition
constraints = [P >> 0]
constraints += [T >> 0]
constraints += [M << -1e-6*np.eye(M.shape[0])]
constraints += [ellip >> 0]

# Objective function definition and problem solving
objective = cp.Minimize(cp.trace(P))

prob = cp.Problem(objective, constraints)

prob.solve(solver=cp.MOSEK, verbose=True)

if prob.status not in ["infeasible", "unbounded"]:
  print(f"Max eigenvalue of P: {np.max(np.linalg.eigvals(P.value))}")
  print(f"Max eigenvalue of M: {np.max(np.linalg.eigvals(M.value))}")