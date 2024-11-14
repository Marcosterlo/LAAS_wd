from systems_and_LMI.systems.NonLinearPendulum import NonLinPendulum
import numpy as np
import cvxpy as cp

# Initialization of the system
s = NonLinPendulum()

# Unpacking of the system parameters
K = s.K
A = s.A
B = s.B
C = s.C
D = s.D
nx = s.nx
nphi = 1
vbar = s.max_torque
Nux = np.zeros((nphi, nx))
Nuw = np.array([[1.0]])
Nvx = K
Nvw = np.array([[0.0]])

# Auxiliary parameters
R = np.linalg.inv(np.eye(Nvw.shape[0]) - Nvw)
Rw = Nux + Nuw @ R @ Nvx
Abar = A + B @ Rw
Bbar = -B @ Nuw @ R

# Variables definition
P = cp.Variable((nx, nx), symmetric=True)
T = cp.Variable((nphi, nphi))
Z = cp.Variable((nphi, nx))

# Parameters definition
alpha = cp.Parameter()

# Constraint matrices definition
Rphi = cp.bmat([
    [np.eye(nx), np.zeros((nx, nphi)), np.zeros((nx, nphi))],
    [R @ Nvx, np.eye(R.shape[0]) - R, np.array([[0.0]])],
    [np.zeros((nphi, nx)), np.eye(nphi), np.array([[0.0]])],
    [np.zeros((nphi, nx)), np.zeros((nphi, nphi)), np.array([[1.0]])]
])

M1 = cp.bmat([
  [np.zeros((nx, nx)), np.zeros((nx, nphi)), np.zeros((nx, nphi)), np.zeros((nx, nphi))],
  [Z, -T , T, np.zeros((nphi, nphi))],
  [np.zeros((nphi, nx)), np.zeros((nphi, nphi)), np.zeros((nphi, nphi)), np.zeros((nphi, nphi))]
])

Sinsec = cp.bmat([
  [0.0, -1.0],
  [-1.0, -2.0]
])

Rs = cp.bmat([
  [np.array([[1.0, 0.0, 0.0]]), np.zeros((1, nphi)), np.zeros((1, nphi))],
  [np.zeros((nphi, nx)), np.zeros((nphi, nphi)), np.array([[1.0]])]
])

ellip = cp.bmat([
  [P, Z.T],
  [Z, 2*alpha*T - alpha**2 * vbar**(-2)]
])

M = cp.bmat([
  [Abar.T @ P @ Abar - P, Abar.T @ P @ Bbar, Abar.T @ P @ C],
  [Bbar.T @ P @ Abar, Bbar.T @ P @ Bbar, Bbar.T @ P @ C],
  [C.T @ P @ Abar, C.T @ P @ Bbar, C.T @ P @ C]
]) - M1 @ Rphi - Rphi.T @ M1.T + Rs.T @ Sinsec @ Rs

# Constraints definition
constraints = [P >> 0]
constraints += [T >> 0]
constraints += [ellip >> 0]
constraints += [M << -1e-8*np.eye(M.shape[0])]

# Objective function definition and problem solution
objective = cp.Minimize(cp.trace(P))

# Problem definition
prob = cp.Problem(objective, constraints)

# Implementation of golden ratio search to find optimal alpha value
feasible_extreme = 1.0
infeasible_extreme = 0.0
threshold = 1e-6
golden_ratio = (1 + np.sqrt(5)) / 2

while (feasible_extreme - infeasible_extreme > threshold):
  x1 = feasible_extreme - (feasible_extreme - infeasible_extreme) / golden_ratio
  x2 = infeasible_extreme + (feasible_extreme - infeasible_extreme) / golden_ratio
  alpha.value = x1
  prob.solve(solver=cp.MOSEK, verbose=False)
  if prob.status in ["infeasible", "unbounded", "unknown"]:
    fx1 = 1e5
  else:
    fx1 = np.max(np.linalg.eigvals(P.value))
    print(f"\n==================== \nMax eigenvalue of P: {fx1}")
    print(f"Max eigenvalue of M: {np.max(np.linalg.eigvals(M.value))}")
    print(f"Current alpha value: {alpha.value}\n==================== \n")
  alpha.value = x2
  prob.solve(solver=cp.MOSEK, verbose=False)
  if prob.status in ["infeasible", "unbounded", "unknown"]:
    fx2 = 1e5
  else:
    fx2 = np.max(np.linalg.eigvals(P.value))
    print(f"\n==================== \nMax eigenvalue of P: {fx2}")
    print(f"Max eigenvalue of M: {np.max(np.linalg.eigvals(M.value))}")
    print(f"Current alpha value: {alpha.value}\n==================== \n")
  if fx1 < fx2:
    feasible_extreme = x2
  else:
    infeasible_extreme = x1

print(f"Max eigenvalue of P: {np.max(np.linalg.eigvals(P.value))}")
print(f"Max eigenvalue of M: {np.max(np.linalg.eigvals(M.value))}")
print(f"Final alpha value: {alpha.value}")