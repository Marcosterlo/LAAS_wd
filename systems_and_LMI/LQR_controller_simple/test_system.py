from systems_and_LMI.systems.LinearPendulum import LinPendulum
import numpy as np
import matplotlib.pyplot as plt

s = LinPendulum()
K = np.load("K.npy")
P = np.load("P.npy")
ellip = np.load("ellip.npy")
max_torque = 1

nsteps = 500

n_trials = 20

for i in range(n_trials):
  states = []
  inputs = []
  lyap = []
  
  in_ellip = False
  while not in_ellip:
    theta0 = np.random.uniform(-20, 20)
    vtheta = np.random.uniform(-20, 20)
    x0 = np.array([[theta0], [vtheta]])
    if x0.T @ P @ x0 <= 1 and x0.T @ P @ x0 >= 0.8:
      print(f'value ellip: {x0.T @ P @ x0}')
      in_ellip = True
      s.state = x0
  
  print(f'Initial state: theta0: {theta0*180/np.pi:.2f}, v0: {vtheta:.2f}')

  for i in range(nsteps):
    u = -K @ s.state
    if u > max_torque:
      u = np.array([[max_torque]])
    elif u < -max_torque:
      u = np.array([[-max_torque]])
    s.state = s.A @ s.state + s.B @ u
    lyap.append((s.state).T @ P @ (s.state))
    states.append(s.state)
    inputs.append(u)
  
  states = np.array(states)
  inputs = np.array(inputs)
  
  # plt.plot(ellip[:, 0], ellip[:, 1], 'ro')
  plt.plot(x0[0], x0[1], 'bo', markersize=10)
  plt.plot(states[:, 0], states[:, 1])
  plt.grid(True)
  plt.show()

  plt.plot(np.arange(0, nsteps), np.squeeze(inputs))
  plt.plot(np.arange(0, nsteps), np.squeeze(lyap))
  plt.grid(True)
  plt.show()