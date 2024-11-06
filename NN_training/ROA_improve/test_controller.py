from systems_and_LMI.systems.NonLinPendulum_no_int_train import NonLinPendulum_no_int_train
import numpy as np
import matplotlib.pyplot as plt
import os

W1_name = os.path.abspath(__file__ + "/../simple_weights/l1.weight.csv")
W2_name = os.path.abspath(__file__ + "/../simple_weights/l2.weight.csv")
W3_name = os.path.abspath(__file__ + "/../simple_weights/l3.weight.csv")
W4_name = os.path.abspath(__file__ + "/../simple_weights/l4.weight.csv")

W1 = np.loadtxt(W1_name, delimiter=',')
W2 = np.loadtxt(W2_name, delimiter=',')
W3 = np.loadtxt(W3_name, delimiter=',')
W4 = np.loadtxt(W4_name, delimiter=',')
W4 = W4.reshape((1, len(W4)))

W = [W1, W2, W3, W4]

b1_name = os.path.abspath(__file__ + "/../simple_weights/l1.bias.csv")
b2_name = os.path.abspath(__file__ + "/../simple_weights/l2.bias.csv")
b3_name = os.path.abspath(__file__ + "/../simple_weights/l3.bias.csv")
b4_name = os.path.abspath(__file__ + "/../simple_weights/l4.bias.csv")

b1 = np.loadtxt(b1_name, delimiter=',')
b2 = np.loadtxt(b2_name, delimiter=',')
b3 = np.loadtxt(b3_name, delimiter=',')
b4 = np.loadtxt(b4_name, delimiter=',')

b = [b1, b2, b3, b4] 

s = NonLinPendulum_no_int_train(W, b)

P_name = os.path.abspath(__file__ + "/../Test/P.npy")
P = np.load(P_name)

in_ellip = False
while not in_ellip:
  theta = np.random.uniform(-np.pi/2, np.pi/2)
  vtheta = np.random.uniform(-s.max_speed, s.max_speed)
  x0 = np.array([[theta], [vtheta]])
  if (x0 - s.xstar).T @ P @ (x0 - s.xstar) < 1:
    in_ellip = True
    print(f"Initial state: theta0 = {theta*180/np.pi:.2f}, vtheta0 = {vtheta:.2f}")
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
inputs = np.array(inputs)
lyap = np.array(lyap)
timegrid = np.arange(0, nsteps+1)

plt.plot(timegrid, states[:, 0])
plt.grid(True)
plt.show()

plt.plot(timegrid, states[:, 1])
plt.grid(True)
plt.show()

plt.plot(timegrid, np.squeeze(inputs))
plt.grid(True)
plt.show()

plt.plot(timegrid, np.squeeze(lyap))
plt.grid(True)
plt.show()