from systems_and_LMI.systems.NonLinearPendulum_NN import NonLinPendulum_NN
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

ref = 0.0

s = NonLinPendulum_NN(ref)
P = np.load('P.npy')
# ellip = np.load('../non_linear_integrator_1_layer/3layerellip.npy')

nsteps = 500
n_trials = 1

for i in range(n_trials):
  states = []
  inputs = []
  lyap = []

  in_ellip = False
  
  while not in_ellip:
    theta0 = np.random.uniform(-np.pi/2, np.pi/2)
    vtheta0 = np.random.uniform(-s.max_speed, s.max_speed)
    x0 = np.array([[theta0], [vtheta0], [0.0]])
    if ((x0).T @ P @ (x0) <= 1):
      in_ellip = True
      
  print(f'Initial state: theta_0: {theta0*180/np.pi:.2f}, v_0: {vtheta0:.2f}, eta_0: {0:.2f}')
  print(f'Is initial point inside ROA? {((x0).T @ P @ (x0) <= 1)[0][0]}')

  s.state = x0
  states.append(x0)

  for i in range(nsteps):
    state, u = s.step()
    states.append(state)
    inputs.append(u)
    lyap.append((state - s.xstar).T @ P @ (state - s.xstar))
  
  states = np.array(states)
  inputs = np.array(inputs)
  lyap = np.array(lyap)
  
  # plt.plot(ellip[:, 0], ellip[:, 1], 'o')
  plt.plot(x0[0], x0[1], 'bo', markersize=10)
  plt.plot(states[:, 0], states[:, 1])
  plt.plot(s.xstar[0], s.xstar[1], 'ro', markersize=10)
  plt.xlabel('Theta')
  plt.xlabel('Vtheta')
  plt.title('Reduced state trajectory')
  plt.grid(True)
  plt.show()

  # plt.plot(np.arange(0, nsteps + 1), states[:, 0] - s.xstar[0])
  # plt.xlabel('Time steps')
  # plt.xlabel('Positon')
  # plt.title('Position plot')
  # plt.grid(True)
  # plt.show()

  # plt.plot(np.arange(0, nsteps + 1), states[:, 1] - s.xstar[1])
  # plt.xlabel('Time steps')
  # plt.xlabel('Velocity')
  # plt.title('Velocity plot')
  # plt.grid(True)
  # plt.show()

  # plt.plot(np.arange(0, nsteps + 1), states[:, 2] - s.xstar[2])
  # plt.xlabel('Time steps')
  # plt.xlabel('Eta evolution')
  # plt.title('Integrator plot')
  # plt.grid(True)
  # plt.show()

  plt.plot(np.arange(0, nsteps), np.squeeze(inputs), label='Control input')
  plt.plot(np.arange(0, nsteps), np.squeeze(lyap), label='Lyapunov function')
  plt.xlabel('Time steps')
  plt.ylabel('Values')
  plt.title('Control input and Lyapunov function')
  plt.legend()
  plt.grid(True)
  plt.show()

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  # ax.plot(ellip[:, 0], ellip[:, 1], ellip[:, 2], 'o')
  ax.plot(x0[0], x0[1], x0[2], 'o', markersize=10)
  ax.plot(s.xstar[0], s.xstar[1], s.xstar[2], 'o', markersize=10)
  ax.plot(states[:, 0], states[:, 1], states[:, 2])
  ax.set_xlabel('Theta')
  ax.set_ylabel('Vtheta')
  ax.set_zlabel('Eta')
  ax.set_title('3D plot of ellip')
  plt.show()