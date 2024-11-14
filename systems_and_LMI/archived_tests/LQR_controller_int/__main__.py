from systems_and_LMI.systems.LinearPendulum_integrator import LinPendulumIntegrator
import numpy as np
import control
import matplotlib.pyplot as plt

s = LinPendulumIntegrator()

A = s.A
B = s.B 

Q = np.array([[100, 0, 0],
              [0, 10, 0],
              [0, 0, 1]])

R = np.array([[1]])

K, _, _ = control.dlqr(A, B, Q, R)

print(f"K: {K}")

eigvals = np.linalg.eigvals(A - B @ K)
print(f"Eigvals of closed loop system: {eigvals}")

stable = all(abs(eig) < 1 for eig in eigvals)
if stable:
  print("System is stable")
else:
  print("System is not stable")

x = np.array([[1.0], [1.0], [0.0]])
states = []
inputs = []

vbar = 10000
for i in range(300):
  u = -K @ x
  if u > vbar:
    u = np.array([[vbar]])
  elif u < -vbar:
    u = np.array([[-vbar]])

  x = A @ x + B @ u
  states.append(x)
  inputs.append(u)

states = np.array(states)

plt.plot(np.arange(0, 300), states[:, 0])
plt.grid(True)
plt.show()

plt.plot(np.arange(0, 300), states[:, 1])
plt.grid(True)
plt.show()

plt.plot(np.arange(0, 300), states[:, 2])
plt.grid(True)
plt.show()

plt.plot(np.arange(0, 300), np.squeeze(inputs))
plt.grid(True)
plt.show()
