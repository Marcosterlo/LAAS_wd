from Recap.system import System
import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

# Weights and biases import
W1_name = os.path.abspath(__file__ + "/../../weights/W1.csv")
W2_name = os.path.abspath(__file__ + "/../../weights/W2.csv")
W3_name = os.path.abspath(__file__ + "/../../weights/W3.csv")
W4_name = os.path.abspath(__file__ + "/../../weights/W4.csv")

b1_name = os.path.abspath(__file__ + "/../../weights/b1.csv")
b2_name = os.path.abspath(__file__ + "/../../weights/b2.csv")
b3_name = os.path.abspath(__file__ + "/../../weights/b3.csv")
b4_name = os.path.abspath(__file__ + "/../../weights/b4.csv")

W1 = np.loadtxt(W1_name, delimiter=',')
W2 = np.loadtxt(W2_name, delimiter=',')
W3 = np.loadtxt(W3_name, delimiter=',')
W4 = np.loadtxt(W4_name, delimiter=',')
W4 = W4.reshape((1, len(W4)))

W = [W1, W2, W3, W4]

b1 = np.loadtxt(b1_name, delimiter=',')
b2 = np.loadtxt(b2_name, delimiter=',')
b3 = np.loadtxt(b3_name, delimiter=',')
b4 = np.loadtxt(b4_name, delimiter=',')

b = [b1, b2, b3, b4]

P0 = np.load('../parameters_search/lambda_0/P.npy')
path0 = "../parameters_search/lambda_0"

# Initial ETM matrices import
bigX1 = np.load(path0 + '/bigX1.npy')
bigX2 = np.load(path0 + '/bigX2.npy')
bigX3 = np.load(path0 + '/bigX3.npy')
bigX4 = np.load(path0 + '/bigX4.npy')
bigX0 = [bigX1, bigX2, bigX3, bigX4]

s = System(W, b, bigX0, 0.0, path0)
ref_bound = 1 * np.pi / 180

nsimulations = 50
initial_states = []
disturbances = []
eta0s = []
for i in range(nsimulations):
  in_ellip = False
  while not in_ellip:
    theta = np.random.uniform(-np.pi/2, np.pi/2)
    vtheta = np.random.uniform(-s.max_speed, s.max_speed)
    ref = np.random.uniform(-ref_bound, ref_bound)*0
    # Initial state definition and system initialization
    x0 = np.array([[theta], [vtheta], [0.0]])
    s = System(W, b, bigX0, ref, path0)
    # Check if the initial state is inside the ellipsoid
    if (x0 - s.xstar).T @ P0 @ (x0 - s.xstar) <= 1.0:
      in_ellip = True
      # Initial eta0 computation with respect to the initial state
      eta0 = ((1 - (x0 - s.xstar).T @ P0 @ (x0 - s.xstar)) / (s.nlayers * 2))[0][0]*0
      eta0s.append(eta0)
      initial_states.append(x0)
      disturbances.append(ref)

nsteps = 1000
nlambda = 50
max_events = nsteps * s.nphi

update_rates_mean = []
update_rates_std = []

ROAs = []

for lam in range(nlambda):
  print(f'Lambda: {lam}')
  update_rate_lam = []
  path = f'../parameters_search/lambda_{lam}'
  # Initial ETM matrices import
  bigX1 = np.load(path + '/bigX1.npy')
  bigX2 = np.load(path + '/bigX2.npy')
  bigX3 = np.load(path + '/bigX3.npy')
  bigX4 = np.load(path + '/bigX4.npy')
  bigX = [bigX1, bigX2, bigX3, bigX4]

  P = np.load(path + '/P.npy')
  ROAs.append(np.pi / np.sqrt(np.linalg.det(P)))
    
ROAs = np.array(ROAs).squeeze()
np.save('ROAs.npy', ROAs)

  #   for sim in range(nsimulations):
  #     s = System(W, b, bigX, disturbances[sim], path)
  #     s.state = initial_states[sim]
  #     s.eta = np.ones(s.nlayers) * eta0s[sim]
  #     n_events = 0
  #     for k in range(nsteps):
  #       _, _, e, _ = s.step()
  #       n_events += e[0] * s.neurons[0] + e[1] * s.neurons[1] + e[2] * s.neurons[2] + e[3] * s.neurons[3]
  #     update_rate = n_events / max_events
  #     update_rate_lam.append(update_rate)
    
  #   update_rates_mean.append(np.mean(update_rate_lam))
  # except(KeyboardInterrupt):
  #   print("Early interruption")
  #   nlambda = lam-1
  #   break

# update_rates_mean = (update_rates_mean - np.min(update_rates_mean)) / (np.max(update_rates_mean) - np.min(update_rates_mean))
# computational_save = -update_rates_mean + (np.max(update_rates_mean) + np.min(update_rates_mean))

import matplotlib.pyplot as plt

value_grid = np.arange(0, nlambda)

# # Apply Gaussian filter for smoothing
# computational_save_smooth = gaussian_filter1d(computational_save, sigma=2)

# interp_func = interp1d(value_grid, computational_save_smooth, kind='cubic')
# computational_save_interp = interp_func(value_grid)
# computational_save_final = (computational_save_interp - np.min(computational_save_interp)) / (np.max(computational_save_interp) - np.min(computational_save_interp)) * (np.max(ROAs) - np.min(ROAs)) + np.min(ROAs)

plt.plot(value_grid, update_rates_mean, label='Computational save')
# plt.plot(value_grid, computational_save, label='Computational save')
# plt.plot(value_grid, ROAs, label='ROA')
# # plt.plot(value_grid, computational_save_interp, label='Computational save (interpolated)')
# # plt.plot(value_grid, computational_save_final, label='Computational save (interpolated)')
plt.xlabel('Lambda')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()