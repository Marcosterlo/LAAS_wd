from systems_and_LMI.systems.NonLinearPendulum import NonLinPendulum
import numpy as np
import matplotlib.pyplot as plt

s = NonLinPendulum()
P = np.load('P.npy')

# Script to uncomment only when you want to generate the ellipsoids sets

# Pbig = np.load('linP.npy')
# P_3 = np.load('../non_linear_integrator_3_layers/P.npy')
# ellipsmall = []
# ellipbig = []
# ellip_3layer = []
# for i in range(10000000):
#   x0 = np.random.uniform(-np.pi, np.pi)
#   v0 = np.random.uniform(-s.max_speed, s.max_speed)
#   x0 = np.array([[x0], [v0], [0.0]])
#   if (x0.T @ P @ x0 <= 1):
#     ellipsmall.append(x0)
#   if (x0.T @ Pbig @ x0 <= 1):
#     ellipbig.append(x0)
#   if (x0.T @ P_3 @ x0 <= 1):
#     ellip_3layer.append(x0)
# ellipsmall = np.array(ellipsmall)
# ellipbig = np.array(ellipbig)
# ellip_3layer = np.array(ellip_3layer)
# np.save('ellip.npy', ellipsmall)
# np.save('linellip.npy', ellipbig)
# np.save('3layerellip.npy', ellip_3layer)

# Import the ellipsoids sets for the non linear and linear system
ellip = np.load('ellip.npy')
linellip = np.load('linellip.npy')
ellip_3layer = np.load('3layerellip.npy')

# Parameters unpacking
max_torque = s.max_torque
s.constant_reference = 0
s.xstar = np.array([[s.constant_reference], [0.0], [0.0]])

# Number of steps for a run and number of episodes to test
nsteps = 500
n_trials = 5

for i in range(n_trials):
  states = []
  inputs = []
  lyap = []
  
  # Random initial state
  theta0 = np.random.uniform(-np.pi/2, np.pi/2)
  vtheta0 = np.random.uniform(-s.max_speed, s.max_speed)
  x0 = np.array([[theta0], [vtheta0], [0.0]])

  print(f"Initial state: theta_0: {theta0*180/np.pi:.2f}, v_0: {vtheta0:.2f}, eta_0: {0:.2f}")
  print(f"Is initial point inside ROA? {(x0.T @ P @ x0 <= 1)[0][0]}")

  s.state = x0
  states.append(x0)

  # Run the system
  for i in range(nsteps):
    u = s.K @ s.state 

    # Perform saturation
    if u > s.max_torque:
      u = np.array([[s.max_torque]])
    elif u < -s.max_torque:
      u = np.array([[-s.max_torque]])

    # Store the state, input and lyapunov function
    state = s.step(u)
    states.append(state)
    inputs.append(u)
    lyap.append((s.state - s.xstar).T @ P @ (s.state - s.xstar))

  states = np.array(states)
  inputs = np.array(inputs)
  lyap = np.array(lyap)

  # Plot non linear ellipsoid along with state trajectory
  plt.plot(ellip[:, 0], ellip[:, 1], 'o')
  plt.plot(x0[0], x0[1], 'bo', markersize=10)
  plt.plot(states[:, 0], states[:, 1])
  plt.xlabel('Theta')
  plt.ylabel('vtheta')
  plt.title('Reduced state trajectory')
  plt.grid(True)
  plt.show()

  # Plot control input and lyapunov function in comparison. For values really far from the ROA we can observe the non strictly decreasing behavior of the Lyapunov function
  plt.plot(np.arange(0, nsteps), np.squeeze(inputs), label='Control input')
  plt.plot(np.arange(0, nsteps), np.squeeze(lyap), label='Lyapunov Function')
  plt.xlabel('Time steps')
  plt.ylabel('Values')
  plt.title('Control input and Lyapunov Function')
  plt.legend()
  plt.grid(True)
  plt.show()

  # Uncomment to plot the linear ellipsoid along with the non linear ellipsoid in comparison

  plt.plot(linellip[:, 0], linellip[:, 1], 'o', label="ROA of linear system")
  plt.plot(ellip[:, 0], ellip[:, 1], 'o', label="ROA of non-linear system")
  plt.ellip(ellip_3layer[:, 0], ellip_3layer[:, 1], 'o', label="ROA of 3 layer non-linear system")
  plt.xlabel('Theta')
  plt.ylabel('vtheta')
  plt.title('Reduced state trajectory')
  plt.grid(True)
  plt.legend()
  plt.show()