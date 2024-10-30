from systems_and_LMI.systems.NonLinPendulum_no_int import NonLinPendulum_no_int
import numpy as np
import matplotlib.pyplot as plt

s = NonLinPendulum_no_int()
theta = np.random.uniform(-np.pi/2, np.pi/2)
vtheta = np.random.uniform(-s.max_speed, s.max_speed)
x0 = np.array([[theta], [vtheta]])
print(f"Initial state: theta0 = {theta}, vtheta0 = {vtheta}")
s.state = x0

states = []
inputs = []

nsteps = 300

for i in range(nsteps):
  state, u = s.step() 
  states.append(state)
  inputs.append(u)
  
states = np.array(states)
inputs = np.array(inputs)
timegrid = np.arange(0, nsteps)

plt.plot(timegrid, states[:, 0])
plt.grid(True)
plt.show()

plt.plot(timegrid, states[:, 1])
plt.grid(True)
plt.show()

plt.plot(timegrid, np.squeeze(inputs))
plt.grid(True)
plt.show()