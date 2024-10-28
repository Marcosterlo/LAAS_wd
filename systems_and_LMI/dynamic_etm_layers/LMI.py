import cvxpy as cp
from system import System
import numpy as np
from scipy.linalg import block_diag

# System initialization
s = System()

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

# Variables for LMI
P = cp.Variable((nx, nx), symmetric=True)
constraints = [P >> 0]

T_val = cp.Variable(nphi)
T = cp.diag(T_val)
constraints += [T >> 0]

Z1 = cp.Variable((neurons, nx))
Z2 = cp.Variable((neurons, nx))
Z = cp.vstack([Z1, Z2])

rho = cp.Variable()
constraints += [rho >= 0]

# Fixed parameters
alpha = 9*1e-4
P0 = np.array([[0.2916, 0.0054], [0.0054, 0.0090]])
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
# constraints += [inclusion >> 0]

# Ellipsoid condition
alpha = cp.Parameter()
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
r1 = 0.6
l1 = decay_rate_eta_1 - r1
gamma1 = (l1 - 1) / r1
gamma1 = np.ones(neurons)*gamma1

decay_rate_eta_2 = 0.9
r2 = 0.4
l2 = decay_rate_eta_2 - r2
gamma2 = (l2 - 1) / r2
gamma2 = np.ones(neurons)*gamma2

gammavec = np.concatenate([gamma1, gamma2], axis=0)

gamma = cp.diag(gammavec)

mat = cp.bmat([
    [np.zeros((nx, nx)), np.zeros((nx, nphi)), np.zeros((nx, nphi))],
    [gamma @ Z, -gamma @ T, gamma @ T]
])

M = cp.vstack([Abar.T, (-B @ Nuw @ R).T]) @ P @ cp.hstack([Abar, -B @ Nuw @ R]) - cp.bmat([
    [P, np.zeros((nx, nphi))],
    [np.zeros((nphi, nx)), np.zeros((nphi, nphi))]
]) + Rphi.T @ mat.T + mat @ Rphi


constraints += [M << -1e-6 * np.eye(M.shape[0])]
constraints += [M + rho * np.eye(M.shape[0]) >> 0]

# Optimization condition
objective = cp.Minimize(rho)

# Problem definition
prob = cp.Problem(objective, constraints)

# Initialization of parameter alpha to 1, the most conservative case 
alpha.value = 1
# Initialization of the variables used to perform the bisection
last_bad = 0
last_good = alpha.value
# Initialization of the variable used to compare the goodness of the solutions
last_P_eig = 1e5
# Flag variable to chck if the run has terminated due to an error
error = False

try:
  # Iter many times to perform the bisection trying to maximize the ROA
  for i in range(100):

    # Problem solution
    prob.solve(solver=cp.MOSEK, verbose=False)

    # Feasible solution
    if prob.status not in ["infeasible", "unbounded", "unknown"]:
      # Maximum eigenvalue of P is smaller the bigger the ROA is
      P_eig = np.max(np.linalg.eigvals(P.value))
      # If the solution is worse than the previous one
      if P_eig > last_P_eig:
        print(f"Feasible but smaller ROA, alpha: {alpha.value}")
        last_bad = alpha.value
        alpha.value = alpha.value + (last_good - last_bad)/8
      else:
        print(f"\n ==================== \nMax eigenvalue of P: {P_eig}")
        print(f"Max eigenvalue of M: {np.max(np.linalg.eigvals(M.value))}")
        print(f"Current alpha value: {alpha.value}\n ==================== \n")
        last_good = alpha.value
        last_P_eig = P_eig
        alpha.value = alpha.value - (last_good - last_bad)/8
        last_P = P.value
        last_M = M.value
    else:
      print(f"Infeasible or unbounded, alpha: {alpha.value}")
      last_bad = alpha.value
      alpha.value = alpha.value + (last_good - last_bad)/8

except cp.error.SolverError:
  error = True
  print("Solver encountered an error, retrieving last feasible solution")
  print(f"Max eigenvalue of P: {np.max(np.linalg.eigvals(last_P))}")
  print(f"Max eigenvalue of M: {np.max(np.linalg.eigvals(last_M))}")
  print(f"Final alpha value: {last_good}")

if not error:
  print(f"\n Max n of iterations reached, retrieving last feasible solution")
  print(f"Max eigenvalue of P: {np.max(np.linalg.eigvals(last_P))}")
  print(f"Max eigenvalue of M: {np.max(np.linalg.eigvals(last_M))}")
  print(f"Final alpha value: {last_good}")