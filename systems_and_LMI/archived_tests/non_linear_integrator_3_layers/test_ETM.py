from systems_and_LMI.systems.NonLinearPendulum_NN_ETM import NonLinPendulum_NN_ETM
import numpy as np
import matplotlib.pyplot as plt

s = NonLinPendulum_NN_ETM()
nsteps = 300
P = np.load('P.npy')

states = []
inputs = []
events = []
etas = []
lyap = []

in_ellip = False
while not in_ellip:
  theta0 = np.random.uniform(-np.pi/2, np.pi/2)
  vtheta0 = np.random.uniform(-s.max_speed, s.max_speed)
  x0 = np.array([[theta0], [vtheta0], [0.0]])
  ref = np.random.uniform(-0.1, 0.1)
  s = NonLinPendulum_NN_ETM(ref)
  if (x0 - s.xstar).T @ P @ (x0 - s.xstar) <= 1:
    in_ellip = True
    s.state = x0
  
# Initial state that is guaranteed to behave nicely
# theta0 = 0.07
# vtheta0 = -0.64
# x0 = np.array([[theta0], [vtheta0], [0.0]])
# s.state = x0
# s.constant_reference = 0.01
print(f"Initial state: theta_0: {theta0:.2f}, vtheta_0: {vtheta0:.2f}, constant disturbance: {ref:.2f}")

for i in range(nsteps):
  state, u, e, eta = s.step() 
  states.append(state)
  inputs.append(u)
  events.append(e)
  lyap.append((state - s.xstar).T @ P @ (state - s.xstar) + 2*eta[0] + 2*eta[1] + 2*eta[2])
  etas.append(eta)

states = np.array(states)
inputs = np.array(inputs)
etas = np.array(etas)
events = np.array(events)
lyap = np.array(lyap)

layer1_trigger = np.sum(events[:, 0]) / nsteps * 100
layer2_trigger = np.sum(events[:, 1]) / nsteps * 100
layer3_trigger = np.sum(events[:, 2]) / nsteps * 100

print(f"Layer 1 has been triggered {layer1_trigger:.1f}% of times")
print(f"Layer 2 has been triggered {layer2_trigger:.1f}% of times")
print(f"Layer 3 has been triggered {layer3_trigger:.1f}% of times")

for i, event in enumerate(events):
  if not event[0]:
    events[i][0] = None
  if not event[1]:
    events[i][1] = None
  if not event[2]:
    events[i][2] = None

fig, axs = plt.subplots(4, 1)
axs[0].plot(np.arange(0, nsteps), np.squeeze(inputs), label='Control input')
axs[0].plot(np.arange(0, nsteps), np.squeeze(inputs)*events[:, 2], marker='o', markerfacecolor='none')
axs[0].set_xlabel('Time steps')
axs[0].set_ylabel('Values')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(np.arange(0, nsteps), np.squeeze(states[:, 0] - s.xstar[0]), label='Position')
axs[1].plot(np.arange(0, nsteps), np.squeeze(states[:, 0] - s.xstar[0])*events[:, 2], marker='o', markerfacecolor='none')
axs[1].set_xlabel('Time steps')
axs[1].set_ylabel('Values')
axs[1].legend()
axs[1].grid(True)

axs[2].plot(np.arange(0, nsteps), np.squeeze(states[:, 1] - s.xstar[1]), label='Velocity')
axs[2].plot(np.arange(0, nsteps), np.squeeze(states[:, 1] - s.xstar[1])*events[:, 2], marker='o', markerfacecolor='none')
axs[2].set_xlabel('Time steps')
axs[2].set_ylabel('Values')
axs[2].legend()
axs[2].grid(True)

axs[3].plot(np.arange(0, nsteps), np.squeeze(states[:, 2] - s.xstar[2]), label='Integrator state')
axs[3].plot(np.arange(0, nsteps), np.squeeze(states[:, 2] - s.xstar[2])*events[:, 2], marker='o', markerfacecolor='none')
axs[3].set_xlabel('Time steps')
axs[3].set_ylabel('Values')
axs[3].legend()
axs[3].grid(True)
plt.show()

plt.plot(np.arange(0, nsteps), np.squeeze(etas[:, 0]), label='Eta_1')
plt.plot(np.arange(0, nsteps), np.squeeze(etas[:, 1]), label='Eta_2')
plt.plot(np.arange(0, nsteps), np.squeeze(etas[:, 2]), label='Eta_3')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(np.arange(0, nsteps), np.squeeze(lyap), label='Lyapunov function')
plt.legend()
plt.grid(True)
plt.show()