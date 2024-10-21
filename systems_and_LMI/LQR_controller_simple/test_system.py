from systems_and_LMI.systems.LinearPendulum import LinPendulum
import numpy as np
import matplotlib.pyplot as plt

s = LinPendulum()
K = np.load("K.npy")
max_torque = s.max_torque

nsteps = 500

n_trials = 20

for i in range(n_trials):
  states = []
  inputs = []
  
  theta0 = np.random.uniform(-np.pi, np.pi)
  vtheta = np.random.uniform(-s.max_speed, s.max_speed)
  x0 = np.array([[theta0], [vtheta]])
  s.state = x0
  states.append(x0)
  
  print(f'Initial state: theta0: {theta0*180/np.pi:.2f} [deg], v0: {vtheta:.2f} [rad/s]')

  for i in range(nsteps):
    u = -K @ s.state
    if u > max_torque:
      u = np.array([[max_torque]])
    elif u < -max_torque:
      u = np.array([[-max_torque]])
    s.state = s.A @ s.state + s.B @ u
    states.append(s.state)
    inputs.append(u)
  
  states = np.array(states)
  inputs = np.array(inputs)
  
  plt.plot(x0[0], x0[1], 'bo', markersize=10)
  plt.plot(states[:, 0], states[:, 1])
  plt.xlabel('Theta')
  plt.ylabel('Vtheta')
  plt.title('State space')
  plt.grid(True)
  plt.show()

  plt.plot(np.arange(0, nsteps), np.squeeze(inputs))
  plt.title('Control inputs')
  plt.grid(True)
  plt.show()