from system import System
import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

# Weights and biases import
W1_name = os.path.abspath(__file__ + "/../weights/W1.csv")
W2_name = os.path.abspath(__file__ + "/../weights/W2.csv")
W3_name = os.path.abspath(__file__ + "/../weights/W3.csv")
W4_name = os.path.abspath(__file__ + "/../weights/W4.csv")

b1_name = os.path.abspath(__file__ + "/../weights/b1.csv")
b2_name = os.path.abspath(__file__ + "/../weights/b2.csv")
b3_name = os.path.abspath(__file__ + "/../weights/b3.csv")
b4_name = os.path.abspath(__file__ + "/../weights/b4.csv")

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

update_rates = []
ROAs = []

nsteps = 300

for i in range(50):
  path = f'parameters_search/lambda_{i}'

  # ETM matrices import
  bigX1 = np.load(path + '/bigX1.npy')
  bigX2 = np.load(path + '/bigX2.npy')
  bigX3 = np.load(path + '/bigX3.npy')
  bigX4 = np.load(path + '/bigX4.npy')
  bigX = [bigX1, bigX2, bigX3, bigX4]

  P = np.load(path + '/P.npy')

  ROAs.append(np.pi/np.sqrt(np.linalg.det(P)))

  s = System(W, b, bigX, 1*np.pi/180, path)

  s.state = np.array([[5.0 * np.pi / 180], [-0.5], [0.0]])
  eta0 = ((1 - (s.state - s.xstar).T @ P @ (s.state - s.xstar)) / (s.nlayers * 2))[0][0]
  # s.eta = np.ones(s.nlayers) * eta0

  events = []

  for i in range(nsteps):
    _, _, e, _ = s.step()
    events.append(e)
  
  events = np.squeeze(np.array(events))
  layer1_trigger = np.sum(events[:, 0]) / nsteps * 100
  layer2_trigger = np.sum(events[:, 1]) / nsteps * 100
  layer3_trigger = np.sum(events[:, 2]) / nsteps * 100
  layer4_trigger = np.sum(events[:, 3]) / nsteps * 100

  update_rates.append((layer1_trigger*s.neurons[0] + layer2_trigger*s.neurons[1] + layer3_trigger*s.neurons[2] + layer4_trigger*s.neurons[3])/ s.nphi)

ROAs = np.array(ROAs).squeeze()
update_rates = np.array(update_rates).squeeze()
pdate_rates = (update_rates - np.min(update_rates)) / (np.max(update_rates) - np.min(update_rates)) * (np.max(ROAs) - np.min(ROAs)) + np.min(ROAs)
computational_save = -update_rates + (np.max(update_rates) + np.min(update_rates))
# Linear interpolation of update_rates

import matplotlib.pyplot as plt

value_grid = np.arange(0, 50)

# Apply Gaussian filter for smoothing
computational_save_smooth = gaussian_filter1d(update_rates, sigma=2)

interp_func = interp1d(value_grid, computational_save_smooth, kind='cubic')
computational_save_interp = interp_func(value_grid)

# update_rates_interp = (update_rates_interp - np.min(update_rates_interp)) / (np.max(update_rates_interp) - np.min(update_rates_interp)) * np.max(ROAs)

plt.plot(value_grid, computational_save, label='Computational save')
plt.plot(value_grid, ROAs, label='ROA')
plt.plot(value_grid, computational_save_interp, label='Computational save (interpolated)')
plt.xlabel('Lambda')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()