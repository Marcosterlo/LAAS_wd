from systems_and_LMI.systems.NonLinPendulum_kETM_train_sat import NonLinPendulum_kETM_train_sat
import os
import numpy as np
import matplotlib.pyplot as plt
from systems_and_LMI.user_defined_functions.ellipsoid_plot_3D import ellipsoid_plot_3D
from systems_and_LMI.user_defined_functions.ellipsoid_plot_2D import ellipsoid_plot_2D

W1_name = os.path.abspath(__file__ + "/../new_weights/mlp_extractor.policy_net.0.weight.csv")
W2_name = os.path.abspath(__file__ + "/../new_weights/mlp_extractor.policy_net.2.weight.csv")
W3_name = os.path.abspath(__file__ + "/../new_weights/mlp_extractor.policy_net.4.weight.csv")
W4_name = os.path.abspath(__file__ + "/../new_weights/action_net.weight.csv")

b1_name = os.path.abspath(__file__ + "/../new_weights/mlp_extractor.policy_net.0.bias.csv")
b2_name = os.path.abspath(__file__ + "/../new_weights/mlp_extractor.policy_net.2.bias.csv")
b3_name = os.path.abspath(__file__ + "/../new_weights/mlp_extractor.policy_net.4.bias.csv")
b4_name = os.path.abspath(__file__ + "/../new_weights/action_net.bias.csv")

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

s = NonLinPendulum_kETM_train_sat(W, b, 0.0)

P = np.load('ETM/P.npy')
print(f"Size of ROA: {np.pi/np.sqrt(np.linalg.det(P)):.2f}")
P_static = np.load('static_ETM/P.npy')
print(f"Size of ROA with static ETM: {np.pi/np.sqrt(np.linalg.det(P_static)):.2f}")

# ref_bound = 5 * np.pi / 180
# in_ellip = False
# while not in_ellip:
#   theta = np.random.uniform(-np.pi/2, np.pi/2)
#   vtheta = np.random.uniform(-s.max_speed, s.max_speed)
#   x0 = np.array([[theta], [vtheta], [0.0]])
#   ref = np.random.uniform(-ref_bound, ref_bound)
#   s = NonLinPendulum_kETM_train_sat(W, b, ref)
#   if (x0).T @ P @ (x0) <= 1.0:
#     in_ellip = True
#     print(f"Initial state: theta0 = {theta*180/np.pi:.2f} deg, vtheta0 = {vtheta:.2f} rad/s, constant reference = {ref*180/np.pi:.2f} deg")
#     s.state = x0
x0 = np.array([[np.pi/12], [0.0], [0.0]])
s = NonLinPendulum_kETM_train_sat(W, b, 0.0)
s.state = x0

nsteps = 500

states = []
inputs = []
events = []
etas = []
lyap = []

for i in range(nsteps):
  state, u, e, eta = s.step()
  states.append(state)
  inputs.append(u)
  events.append(e)
  etas.append(eta)
  lyap.append((state - s.xstar).T @ P @ (state - s.xstar) + 2*eta[0] + 2*eta[1] + 2*eta[2] + 2*eta[3])

states = np.insert(states, 0, x0, axis=0)
states = np.delete(states, -1, axis=0)
states = np.squeeze(np.array(states))
states[:, 0] *= 180 / np.pi
s.xstar[0] *= 180 / np.pi

inputs = np.insert(inputs, 0, np.array(0.0), axis=0)
inputs = np.delete(inputs, -1, axis=0)
inputs = np.squeeze(np.array(inputs))

events = np.squeeze(np.array(events))
etas = np.squeeze(np.array(etas))
lyap = np.squeeze(np.array(lyap))

timegrid = np.arange(0, nsteps)

layer1_trigger = np.sum(events[:, 0]) / nsteps * 100
layer2_trigger = np.sum(events[:, 1]) / nsteps * 100
layer3_trigger = np.sum(events[:, 2]) / nsteps * 100
layer4_trigger = np.sum(events[:, 3]) / nsteps * 100

print(f"Layer 1 has been triggered {layer1_trigger:.1f}% of times")
print(f"Layer 2 has been triggered {layer2_trigger:.1f}% of times")
print(f"Layer 3 has been triggered {layer3_trigger:.1f}% of times")
print(f"Output layer has been triggered {layer4_trigger:.1f}% of times")

for i, event in enumerate(events):
  if not event[0]:
    events[i][0] = None
  if not event[1]:
    events[i][1] = None
  if not event[2]:
    events[i][2] = None
  if not event[3]:
    events[i][3] = None
    
fig, axs = plt.subplots(4, 1)
axs[0].plot(timegrid, inputs, label='Control input')
axs[0].plot(timegrid, inputs * events[:, 3], marker='o', markerfacecolor='none', linestyle='None')
axs[0].plot(timegrid, timegrid * 0 + s.wstar[-1] * s.max_torque, 'r--')
axs[0].set_xlabel('Time steps')
axs[0].set_ylabel('Values')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(timegrid, states[:, 0], label='Position')
axs[1].plot(timegrid, states[:, 0] * events[:, 3], marker='o', markerfacecolor='none', linestyle='None')
axs[1].plot(timegrid, timegrid * 0 + s.xstar[0], 'r--')
axs[1].set_xlabel('Time steps')
axs[1].set_ylabel('Values')
axs[1].legend()
axs[1].grid(True)

axs[2].plot(timegrid, states[:, 1], label='Velocity')
axs[2].plot(timegrid, states[:, 1] * events[:, 3], marker='o', markerfacecolor='none', linestyle='None')
axs[2].plot(timegrid, timegrid * 0 + s.xstar[1], 'r--')
axs[2].set_xlabel('Time steps')
axs[2].set_ylabel('Values')
axs[2].legend()
axs[2].grid(True)

axs[3].plot(timegrid, states[:, 2], label='Integrator state')
axs[3].plot(timegrid, states[:, 2] * events[:, 3], marker='o', markerfacecolor='none', linestyle='None')
axs[3].plot(timegrid, timegrid * 0 + s.xstar[2], 'r--')
axs[3].set_xlabel('Time steps')
axs[3].set_ylabel('Values')
axs[3].legend()
axs[3].grid(True)
plt.show()

plt.plot(timegrid, etas[:, 0], label='Eta_1')
plt.plot(timegrid, etas[:, 1], label='Eta_2')
plt.plot(timegrid, etas[:, 2], label='Eta_3')
plt.plot(timegrid, etas[:, 3], label='Eta_4')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(timegrid, lyap, label='Lyapunov function')
plt.legend()
plt.grid(True)
plt.show()

fig, ax = ellipsoid_plot_3D(P, False, color='b', legend='ROA with dynamic ETM')
ax.plot(states[:, 0], states[:, 1], states[:, 2], 'b')
ax.plot(s.xstar[0], s.xstar[1], s.xstar[2], marker='o', markersize=5, color='r')
ax = ellipsoid_plot_3D(P_static, False, ax, color='r', legend='ROA with static ETM')
plt.legend()
plt.show()

fig, ax = ellipsoid_plot_2D(P[:2, :2], False, color='b', legend='ROA with dynamic ETM')
ax.plot(states[:, 0], states[:, 1], 'b')
ax.plot(s.xstar[0], s.xstar[1], marker='o', markersize=5, color='r')
ax = ellipsoid_plot_2D(P_static[:2, :2], False, ax, color='r', legend='ROA with static ETM')
plt.legend()
plt.show()