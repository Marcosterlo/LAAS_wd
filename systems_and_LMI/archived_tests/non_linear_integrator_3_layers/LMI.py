from systems_and_LMI.systems.NonLinearPendulum_NN import NonLinPendulum_NN
import numpy as np
import cvxpy as cp

# System initialization
s = NonLinPendulum_NN()

# Parameters unpacking
A = s.A
B = s.B
C = s.C
nx = s.nx
nu = s.nu
nphi = s.nphi
vbar = s.bound
neurons = s.neurons
nlayer = s.nlayer
N = s.N
Nux = N[0]
Nuw = N[1]
Nub = N[2]
Nvx = N[3]
Nvw = N[4]
Nvb = N[5]
R = s.R
Rw = s.Rw
Rb = s.Rb
nq = 1

# Auxiliary parameters
Abar = A + B @ Rw
Bbar = -B @ Nuw @ R

# Variables definition
P = cp.Variable((nx, nx), symmetric=True)

T_val = cp.Variable(nphi)
T = cp.diag(T_val)

Z1 = cp.Variable((neurons[0], nx))
Z2 = cp.Variable((neurons[1], nx))
Z3 = cp.Variable((neurons[2], nx))
Z = cp.vstack([Z1, Z2, Z3])

# Constraint matrices definition
Rphi = cp.bmat([
    [np.eye(nx), np.zeros((nx, nphi)), np.zeros((nx, nq))],
    [R @ Nvx, np.eye(R.shape[0]) - R, np.zeros((nphi, nq))],
    [np.zeros((nphi, nx)), np.eye(nphi), np.zeros((nphi, nq))],
    [np.zeros((nq, nx)), np.zeros((nq, nphi)), np.eye(nq)]
])

M1 = cp.bmat([
  [np.zeros((nx, nx)), np.zeros((nx, nphi)), np.zeros((nx, nphi)), np.zeros((nx, nq))],
  [Z, -T , T, np.zeros((nphi, nq))],
  [np.zeros((nq, nx)), np.zeros((nq, nphi)), np.zeros((nq, nphi)), np.zeros((nq, nq))]
])

Sinsec = cp.bmat([
  [0.0, -1.0],
  [-1.0, -2.0]
])

Rs = cp.bmat([
  [np.array([[1.0, 0.0, 0.0]]), np.zeros((1, nphi)), np.zeros((1, nq))],
  [np.zeros((nq, nx)), np.zeros((nq, nphi)), np.array([[1.0]])]
])

M = cp.bmat([
  [Abar.T @ P @ Abar - P, Abar.T @ P @ Bbar, Abar.T @ P @ C],
  [Bbar.T @ P @ Abar, Bbar.T @ P @ Bbar, Bbar.T @ P @ C],
  [C.T @ P @ Abar, C.T @ P @ Bbar, C.T @ P @ C]
]) - M1 @ Rphi - Rphi.T @ M1.T + Rs.T @ Sinsec @ Rs

# Constraints definition
constraints = [P >> 0]
constraints += [T >> 0]
constraints += [M << -1e-6 * np.eye(M.shape[0])]

# Ellipsoid conditions
alpha = cp.Parameter()

for i in range(nlayer - 1):
  for k in range(neurons[i]):
    Z_el = Z[i*neurons[i] + k]
    T_el = T[i*neurons[i] + k, i*neurons[i] + k]
    vcap = np.min([np.abs(-vbar - s.wstar[i][k][0]), np.abs(vbar - s.wstar[i][k][0])], axis=0)
    ellip = cp.bmat([
        [P, cp.reshape(Z_el, (nx ,1))],
        [cp.reshape(Z_el, (1, nx)), cp.reshape(2*alpha*T_el - alpha**2*vcap**(-2), (1, 1))] 
    ])
    constraints += [ellip >> 0]

# Objective function definition and problem solution
objective = cp.Minimize(cp.trace(P))

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
  print(f"\nMax n of iterations reached, retrieving last feasible solution")
  print(f"Max eigenvalue of P: {np.max(np.linalg.eigvals(last_P))}")
  print(f"Max eigenvalue of M: {np.max(np.linalg.eigvals(last_M))}")
  print(f"Final alpha value: {last_good}")