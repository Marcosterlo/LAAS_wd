from systems_and_LMI.systems.NonLinearPendulum_NN_kETM import NonLinPendulum_NN_kETM
from systems_and_LMI.systems.NonLinearPendulum_NN_ETM import NonLinPendulum_NN_ETM
import numpy as np
import matplotlib.pyplot as plt

ref = 0.0
s = NonLinPendulum_NN_kETM(ref)
s1 = NonLinPendulum_NN_ETM(ref)
nsteps = 300
P = np.load('P.npy')
kP = np.load('kP.npy')

states = []
inputs = []
events = []
etas = []
lyap = []

states1 = []
inputs1 = []
events1 = []
etas1 = []
lyap1 = []

in_ellip = False
while not in_ellip:
  theta0 = np.random.uniform(-np.pi/2, np.pi/2)
  vtheta0 = np.random.uniform(-s.max_speed, s.max_speed)
  ref = np.random.uniform(-0.1, 0.1)
  s = NonLinPendulum_NN_kETM(ref)
  s1 = NonLinPendulum_NN_ETM(ref)
  x0 = np.array([[theta0], [vtheta0], [0.0]])
  if (x0 - s.xstar).T @ kP @ (x0 - s.xstar) <= 1 and (x0 - s1.xstar).T @ P @ (x0 - s1.xstar) <= 1:
    in_ellip = True
    s.state = x0
    s1.state = x0

print(f"Initial state: theta_0: {theta0:.2f}, vtheta_0: {vtheta0:.2f}, constant disturbance: {ref:.2f}")

for i in range(nsteps):
  state, u, e, eta = s.step() 
  state1, u1, e1, eta1 = s1.step()
  states.append(state)
  inputs.append(u)
  events.append(e)
  lyap.append((state - s.xstar).T @ kP @ (state - s.xstar) + 2*eta[0] + 2*eta[1] + 2*eta[2])

  etas.append(eta)
  states1.append(state1)
  inputs1.append(u1)
  events1.append(e1)
  lyap1.append((state1 - s1.xstar).T @ P @ (state1 - s1.xstar) + 2*eta1[0] + 2*eta1[1] + 2*eta1[2])
  etas1.append(eta1)

states = np.array(states)
inputs = np.array(inputs)
etas = np.array(etas)
events = np.array(events)
lyap = np.array(lyap)

states1 = np.array(states1)
inputs1 = np.array(inputs1)
etas1 = np.array(etas1)
events1 = np.array(events1)
lyap1 = np.array(lyap1)

layer1_trigger_1 = np.sum(events[:, 0]) / nsteps * 100
layer2_trigger_1 = np.sum(events[:, 1]) / nsteps * 100
layer3_trigger_1 = np.sum(events[:, 2]) / nsteps * 100

print(f"New ETM system:\n")
print(f"Layer 1 has been triggered {layer1_trigger_1:.1f}% of times")
print(f"Layer 2 has been triggered {layer2_trigger_1:.1f}% of times")
print(f"Layer 3 has been triggered {layer3_trigger_1:.1f}% of times")

layer1_trigger_2 = np.sum(events1[:, 0]) / nsteps * 100
layer2_trigger_2 = np.sum(events1[:, 1]) / nsteps * 100
layer3_trigger_2 = np.sum(events1[:, 2]) / nsteps * 100

print(f"\nOld ETM system:\n")
print(f"Layer 1 has been triggered {layer1_trigger_2:.1f}% of times")
print(f"Layer 2 has been triggered {layer2_trigger_2:.1f}% of times")
print(f"Layer 3 has been triggered {layer3_trigger_2:.1f}% of times")

for i, event in enumerate(events):
  if not event[0]:
    events[i][0] = None
  if not event[1]:
    events[i][1] = None
  if not event[2]:
    events[i][2] = None

for i, event in enumerate(events1):
  if not event[0]:
    events1[i][0] = None
  if not event[1]:
    events1[i][1] = None
  if not event[2]:
    events1[i][2] = None

timegrid = np.arange(0, nsteps)

fig, axs = plt.subplots(4, 1)
axs[0].plot(timegrid, np.squeeze(inputs), label='Control input (new ETM)')
axs[0].plot(timegrid, np.squeeze(inputs1), label='Control input (old ETM)')
axs[0].plot(timegrid, np.squeeze(inputs)*events[:,2], marker='o', markerfacecolor='none')
axs[0].plot(timegrid, np.squeeze(inputs1)*events1[:,2], marker='o', markerfacecolor='none')
axs[0].set_xlabel('Time steps')
axs[0].set_ylabel('Values')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(timegrid, np.squeeze(states[:, 0] - s.xstar[0]), label='Position new ETM')
axs[1].plot(timegrid, np.squeeze(states[:, 0] - s.xstar[0])*events[:,2], marker='o', markerfacecolor='none')
axs[1].plot(timegrid, np.squeeze(states1[:, 0] - s1.xstar[0]), label='Position old ETM')
axs[1].plot(timegrid, np.squeeze(states1[:, 0] - s1.xstar[0])*events1[:,2], marker='o', markerfacecolor='none')
axs[1].set_xlabel('Time steps')
axs[1].set_ylabel('Values')
axs[1].legend()
axs[1].grid(True)

axs[2].plot(timegrid, np.squeeze(states[:, 1] - s.xstar[1]), label='Velocity new ETM')
axs[2].plot(timegrid, np.squeeze(states[:, 1] - s.xstar[1])*events[:,2], marker='o', markerfacecolor='none')
axs[2].plot(timegrid, np.squeeze(states1[:, 1] - s1.xstar[1]), label='Velocity old ETM')
axs[2].plot(timegrid, np.squeeze(states1[:, 1] - s1.xstar[1])*events1[:,2], marker='o', markerfacecolor='none')
axs[2].set_xlabel('Time steps')
axs[2].set_ylabel('Values')
axs[2].legend()
axs[2].grid(True)

axs[3].plot(timegrid, np.squeeze(states[:, 2] - s.xstar[2]), label='Integrator state')
axs[3].plot(timegrid, np.squeeze(states[:, 2] - s.xstar[2])*events[:,2], marker='o', markerfacecolor='none')
axs[3].plot(timegrid, np.squeeze(states1[:, 2] - s1.xstar[2]), label='Integrator state new ETM')
axs[3].plot(timegrid, np.squeeze(states1[:, 2] - s1.xstar[2])*events1[:,2], marker='o', markerfacecolor='none')
axs[3].set_xlabel('Time steps')
axs[3].set_ylabel('Values')
axs[3].legend()
axs[3].grid(True)

plt.show()

plt.plot(timegrid, np.squeeze(etas[:, 0]), label='Eta 1')
plt.plot(timegrid, np.squeeze(etas[:, 1]), label='Eta 2')
plt.plot(timegrid, np.squeeze(etas[:, 2]), label='Eta 3')
plt.plot(timegrid, np.squeeze(etas1[:, 0]), label='Eta 1 old')
plt.plot(timegrid, np.squeeze(etas1[:, 1]), label='Eta 2 old')
plt.plot(timegrid, np.squeeze(etas1[:, 2]), label='Eta 3 old')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(timegrid, np.squeeze(lyap), label='Lyapunov function')
plt.plot(timegrid, np.squeeze(lyap1), label='Lyapunov function old etm')
plt.legend()
plt.grid(True)
plt.show()