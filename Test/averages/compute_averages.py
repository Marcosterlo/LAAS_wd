import os
import numpy as np
from Test.system import System

# path = '../static'
# path = '../dynamic'
# path = '../finstatic_noopt'
path = '../finstatic_opt'

# rho = 0.86
rho = 0

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

# New ETM matrices import
bigX1 = np.load(path + '/bigX1.npy')
bigX2 = np.load(path + '/bigX2.npy')
bigX3 = np.load(path + '/bigX3.npy')
bigX4 = np.load(path + '/bigX4.npy')

bigX = [bigX1, bigX2, bigX3, bigX4]

# System initialization
s = System(W, b, bigX, 0.0, path)

P = np.load(path + '/P.npy')
volume = 4/3*np.pi/np.sqrt(np.linalg.det(P))
print(f"Volume of ellipsoid: {volume:.2f}")

init_configs = np.load('init_configs.npy')

# Good rho values for both optimized and non-optimized case:
# STATIC:
# optim rho = 0.00 = 35.62
# nonoptim rho = 0.00 = 37.29
# DYNAMIC:
# rho = 0.70 = optim 33.21 -> nonoptim 33.93
# rho = 0.86 = optim 29.27 -> nonoptim 35.98
# rho = 0.87 = optim 32.04 -> nonoptim 35.68
# rho = 0.89 = optim 33.15 -> nonoptim 35.76

for config in init_configs:
  theta, vtheta, ref, eta0 = config
  x0 = np.array([[theta], [vtheta], [0.0]])
  s = System(W, b, bigX, ref, path)
  s.state = x0
  s.eta = np.ones(s.nlayers) * eta0
  s.rho = np.ones(s.nlayers)*rho
  events = []
  update_rates = []
  lyap = []
  steps = []
  in_loop = True
  # lyap_magnitude = 1e-15
  lyap_magnitude = 1e-40
  max_steps = 349
  # max_steps = 5000
  nsteps = 0

  # Simulation loop
  while in_loop:
    nsteps += 1
    state, _, e, eta = s.step()
    events.append(e)
    if 'static' in path:
      lyap.append((state - s.xstar).T @ P @ (state - s.xstar))
    else:
      lyap.append((state - s.xstar).T @ P @ (state - s.xstar) + 2*eta[0] + 2*eta[1] + 2*eta[2] + 2*eta[3])
    # Stop condition
    if lyap[-1] < lyap_magnitude or nsteps > max_steps:
      in_loop = False

  events = np.array(events).squeeze()
  steps.append(nsteps)
  layer1_trigger = np.sum(events[:, 0]) / nsteps * 100
  layer2_trigger = np.sum(events[:, 1]) / nsteps * 100
  layer3_trigger = np.sum(events[:, 2]) / nsteps * 100
  layer4_trigger = np.sum(events[:, 3]) / nsteps * 100
  total_trigger = (layer1_trigger * s.neurons[0] + layer2_trigger * s.neurons[1] + layer3_trigger * s.neurons[2] + layer4_trigger * s.neurons[3]) / (s.nphi)
  update_rates.append([layer1_trigger, layer2_trigger, layer3_trigger, layer4_trigger, total_trigger])

new_path = path[3:]
steps = np.mean(np.array(steps).squeeze())
mean_rates = np.mean(update_rates, axis=0)
print(new_path + f" average number of steps: {steps:.2f}")
print(new_path + f" layer update rates: {mean_rates[0]:.2f}, {mean_rates[1]:.2f}, {mean_rates[2]:.2f}, {mean_rates[3]:.2f}")
print(new_path + f" total update rate: {mean_rates[-1]:.2f}")