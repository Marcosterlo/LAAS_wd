from systems_and_LMI.systems.LinearPendulum_integrator import LinPendulumIntegrator
import numpy as np
import cvxpy as cp

s = LinPendulumIntegrator()

K = np.load("K.npy")
A = s.A
B = s.B
R = np.array([[1]])
Rw = np.zeros((1, 3))
Rb = np.array([[0]])
Nux = np.zeros((1, 3))
Nuw = np.array([[1]])
Nub = np.array([[0]])
Nvx = K
Nvw = np.array([[0]])
Nvb = np.array([[0]])
nphi = 1

P = cp.Variable((s.nx, s.nx), symmetric=True) 
T = cp.Variable((1, 1))
Z = cp.Variable((1, 3))
trans = cp.bmat([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
])
P_ellip = trans @ P @ trans

Rphi = cp.bmat([
    [np.eye(s.nx), np.zeros((s.nx, nphi))],
    [K, np.array([[0.]])],
    [np.zeros((nphi, s.nx)), np.array([[1.]])]
])

mat = cp.bmat([
    [np.zeros((s.nx, s.nx)), np.zeros((s.nx, nphi)), np.zeros((s.nx, nphi))],
    [Z, -T, T]
])

M = cp.vstack([A.T, -B.T]) @ P @ cp.hstack([A, -B]) - cp.bmat([
    [P, np.zeros((3, 1))],
    [np.zeros((1, 3)), np.array([[0.]])]
]) - Rphi.T @ mat.T - mat @ Rphi

constraints = [P >> 0]
constraints += [T >= 0]
constraints += [M << -1e-4 * np.eye(M.shape[0])]

vbar = 20.0
alpha = 9 * 1e-4

ellip = cp.bmat([
    [P_ellip, Z.T],
    [Z, cp.reshape(2*alpha*T - alpha**2*vbar**(-2), (1, 1))]
])
constraints += [ellip >> 0]

objective = cp.Minimize(cp.trace(P))

prob = cp.Problem(objective, constraints)

prob.solve(solver=cp.SCS, verbose=True)

if prob.status not in ["infeasible", "unbounded"]:
  print(f"Max eigenvalue of P: {np.max(np.linalg.eigvals(P.value))}")
  print(f"Max eigenvalue of M: {np.max(np.linalg.eigvals(M.value))}")