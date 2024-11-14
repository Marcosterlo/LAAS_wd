from systems_and_LMI.systems.LinearPendulum_integrator import LinPendulumIntegrator
import numpy as np
import matplotlib.pyplot as plt

s = LinPendulumIntegrator()
K = np.load("K.npy")
P = np.load("P.npy")
ellip = np.load("ellip.npy")
max_torque = s.max_torque
s.constant_reference = 0
s.xstar = np.array([[s.constant_reference], [0.0], [0.0]])

nsteps = 500
n_trials = 5

for i in range(n_trials):
  states = []
  inputs = []
  lyap = []

  theta0 = np.random.uniform(-np.pi/2, np.pi/2)
  vtheta = np.random.uniform(-s.max_speed, s.max_speed)
  x0 = np.array([[theta0], [vtheta], [0.0]])
  # x0 = np.array([[np.pi], [2.0], [0.0]])
  s.state = x0
  states.append(x0)

  print(f"Initial state: theta0: {theta0*180/np.pi:.2f} [deg], v0: {vtheta:.2f} [rad/s], eta0: {0:.2f}")
  print(f"Is initial point inside ROA? {(x0.T @ P @ x0 <= 1)[0][0]}")

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
  lyap = np.array(lyap)

  plt.plot(ellip[:, 0], ellip[:, 1], 'o')
  plt.plot(x0[0], x0[1], 'bo', markersize=10)
  plt.plot(states[:, 0], states[:, 1])
  plt.xlabel("theta")
  plt.ylabel("vtheta")
  plt.title("Reduced state trajectory")
  plt.grid(True)
  plt.show()

  plt.plot(np.arange(0, nsteps), np.squeeze(inputs), label='Control Input')
  plt.plot(np.arange(0, nsteps), np.squeeze(lyap), label='Lyapunov Function')
  plt.xlabel("Time Steps")
  plt.ylabel("Values")
  plt.title("Control Input and Lyapunov Function")
  plt.legend()
  plt.grid(True)
  plt.show()

  ax = plt.axes(projection='3d')
  plt.plot(states[:, 0], states[:, 1], states[:, 2])
  plt.plot(x0[0], x0[1], x0[2], 'bo', markersize=10)
  plt.plot(0, 0, 0, 'go', markersize=10)
  plt.xlabel("theta")
  plt.ylabel("vtheta")
  ax.set_zlabel("eta")
  plt.title("Full state trajectory")
  plt.show()