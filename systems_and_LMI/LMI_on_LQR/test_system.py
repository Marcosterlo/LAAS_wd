from systems_and_LMI.systems.LinearPendulum_integrator import LinPendulumIntegrator
import numpy as np
import matplotlib.pyplot as plt

s = LinPendulumIntegrator()
K = np.load("K.npy")
P = np.load("P.npy")
max_torque = s.max_torque
s.constant_reference = 0
s.xstar = np.array([[s.constant_reference], [0.0], [0.0]])

ellip = []

for i in range(1000000):
  x0 = np.array([[np.random.uniform(-20, 20)], [np.random.uniform(-20, 20)], [0.0]])
  if ((x0 - s.xstar).T @ P @ (x0 - s.xstar) <= 1):
    ellip.append(x0)

ellip = np.squeeze(np.array(ellip))
np.save('ellip.npy', ellip)

nsteps = 500


n_trials = 20

for i in range(n_trials):
  states = []
  inputs = []
  lyap = []

  in_ellip = False
  while not in_ellip:
    theta0 = np.random.uniform(-np.pi, np.pi)
    vtheta = np.random.uniform(-s.max_speed, s.max_speed)
    x0 = np.array([[theta0], [vtheta], [0.0]])
    if ((x0 - s.xstar).T @ P @ (x0 - s.xstar) <= 1):
      in_ellip = True
      s.state = x0

  print(f"Initial state: theta0: {theta0*180/np.pi:.2f}, v0: {vtheta:.2f}, eta0: {0:.2f}")

  for i in range(nsteps):
    u = -K @ s.state
    if u > max_torque:
      u = np.array([[max_torque]])
    elif u < -max_torque:
      u = np.array([[-max_torque]])
    s.state = s.A @ s.state + s.B @ u
    s.state[2] += -s.constant_reference
    lyap.append((s.state - s.xstar).T @ P @ (s.state - s.xstar))
    states.append(s.state)
    inputs.append(u)


  states = np.array(states)
  inputs = np.array(inputs)

  plt.plot(ellip[:, 0], ellip[:, 1], 'ro')
  plt.plot(x0[0], x0[1], 'bo', markersize=10)
  plt.plot(states[:, 0], states[:, 1])
  plt.grid(True)
  plt.show()

  plt.plot(np.arange(0, nsteps), np.squeeze(inputs))
  plt.plot(np.arange(0, nsteps), np.squeeze(lyap))
  plt.grid(True)
  plt.show()