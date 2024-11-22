from systems_and_LMI.systems.NonLinPendulum_train import NonLinPendulum_train
import numpy as np
import matplotlib.pyplot as plt
import os
from systems_and_LMI.LMI.int_3l.main import LMI_3l_int
from systems_and_LMI.user_defined_functions.ellipsoid_plot_3D import ellipsoid_plot_3D
from systems_and_LMI.user_defined_functions.ellipsoid_plot_2D import ellipsoid_plot_2D

W1_name = os.path.abspath(__file__ + '/../new_weights/mlp_extractor.policy_net.0.weight.csv')
b1_name = os.path.abspath(__file__ + '/../new_weights/mlp_extractor.policy_net.0.bias.csv')
W2_name = os.path.abspath(__file__ + '/../new_weights/mlp_extractor.policy_net.2.weight.csv')
b2_name = os.path.abspath(__file__ + '/../new_weights/mlp_extractor.policy_net.2.bias.csv')
W3_name = os.path.abspath(__file__ + '/../new_weights/mlp_extractor.policy_net.4.weight.csv')
b3_name = os.path.abspath(__file__ + '/../new_weights/mlp_extractor.policy_net.4.bias.csv')
W4_name = os.path.abspath(__file__ + '/../new_weights/action_net.weight.csv')
b4_name = os.path.abspath(__file__ + '/../new_weights/action_net.bias.csv')

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

s = NonLinPendulum_train(W, b, 0.0)

new_weights = False
if new_weights:
  lmi = LMI_3l_int(W, b)
  alpha = lmi.search_alpha(0.2, 0, 1e-5, verbose=True)
  P, _, _ = lmi.solve(alpha, verbose=True)
  lmi.save_results('Test')
else:
  P = np.load('static_ETM/P.npy')
print(f"Size of ROA: {np.pi/np.sqrt(np.linalg.det(P)):.2f}")

in_ellip = False
while not in_ellip:
    theta = np.random.uniform(-np.pi/2, np.pi/2)
    vtheta = np.random.uniform(-s.max_speed, s.max_speed)
    ref_bound = 10 * np.pi / 180
    ref = np.random.uniform(-ref_bound, ref_bound)
    s = NonLinPendulum_train(W, b, ref)
    x0 = np.array([[theta], [vtheta], [0.0]])
    if (x0).T @ P @ (x0) <= 1:
        in_ellip = True
        print(f"Initial state: theta0 = {theta*180/np.pi:.2f} deg, vtheta0 = {vtheta:.2f} rad/s, constant reference = {ref*180/np.pi:.2f} deg")
        s.state = x0
    
states = []
inputs = []
lyap = []

nsteps = 300

states.append(s.state)
inputs.append(np.array([[0.0]]))
lyap.append((s.state - s.xstar).T @ P @ (s.state - s.xstar))

for i in range(nsteps):
  state, u = s.step() 
  states.append(state)
  inputs.append(u)
  lyap.append((state - s.xstar).T @ P @ (state - s.xstar))
  
states = np.array(states)
states[:, 0] = states[:, 0] * 180 / np.pi
s.xstar[0] = s.xstar[0] * 180 / np.pi
inputs = np.array(inputs)
lyap = np.array(lyap)
timegrid = np.arange(0, nsteps+1)

plt.plot(timegrid, states[:, 0])
plt.plot(timegrid, timegrid*0.0 + s.xstar[0], 'r--')
plt.grid(True)
plt.show()

plt.plot(timegrid, states[:, 1])
plt.plot(timegrid, timegrid*0.0 + s.xstar[1], 'r--')
plt.grid(True)
plt.show()

plt.plot(timegrid, states[:, 2])
plt.plot(timegrid, timegrid*0.0 + s.xstar[2], 'r--')
plt.grid(True)
plt.show()

plt.plot(timegrid, np.squeeze(inputs))
plt.grid(True)
plt.show()

plt.plot(timegrid, np.squeeze(lyap))
plt.grid(True)
plt.show()

fix, ax = ellipsoid_plot_3D(P, False)
ax.plot(states[:, 0], states[:, 1], states[:, 2], 'b')
ax.plot(s.xstar[0], s.xstar[1], s.xstar[2], marker='o', markersize=5, color='r')
plt.show()

P = P[:2, :2]
fig, ax = ellipsoid_plot_2D(P, False)
ax.plot(states[:, 0], states[:, 1], 'b')
ax.plot(s.xstar[0], s.xstar[1], marker='o', markersize=5, color='r')
plt.show()