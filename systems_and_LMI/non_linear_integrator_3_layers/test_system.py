from systems_and_LMI.systems.NonLinearPendulum_NN import NonLinPendulum_NN
import numpy as np
import matplotlib.pyplot as plt

s = NonLinPendulum_NN()

nsteps = 500
n_trials = 5

for i in range(n_trials):
  states = []
  inputs = []
  
  theta0 = np.random.uniform(-np.pi/2, np.pi/2)
  vtheta0 = np.random.uniform(-s.max_speed, s.max_speed)
  x0 = np.array([[theta0], [vtheta0], [0.0]])

  print(f'Initial state: theta_0: {theta0*180/np.pi:.2f}, v_0: {vtheta0:.2f}, eta_0: {0:.2f}')

  s.state = x0
  states.append(x0)

  for i in range(nsteps):
    state, u = s.step()
    states.append(state)
    inputs.append(u)
  
  states = np.array(states)
  inputs = np.array(inputs)
  
  plt.plot(x0[0], x0[1], 'bo', markersize=10)
  plt.plot(states[:, 0], states[:, 1])
  plt.xlabel('Theta')
  plt.xlabel('Vtheta')
  plt.title('Reduced state trajectory')
  plt.grid(True)
  plt.show()

  plt.plot(np.arange(0, nsteps), np.squeeze(inputs), label='Control input')
  plt.xlabel('Time steps')
  plt.ylabel('Values')
  plt.title('Control input')
  plt.legend()
  plt.grid(True)
  plt.show()