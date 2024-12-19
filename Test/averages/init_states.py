import numpy as np  
from Test.system import System
import os

path = '../dynamic'

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

print(f"Volume of ellipsoid: {4/3*np.pi/np.sqrt(np.linalg.det(P)):.2f}")

# Maximum disturbance bound on the position theta in degrees
ref_bound = 5 * np.pi / 180

initial_configs = []

# Loop to find a random initial state inside the ellipsoid

for i in range(100):
  in_ellip = False
  while not in_ellip:
    # Random initial state and disturbance
    theta = np.random.uniform(-np.pi/2, np.pi/2)
    vtheta = np.random.uniform(-s.max_speed, s.max_speed)
    ref = np.random.uniform(-ref_bound, ref_bound)

    # Initial state definition and system initialization
    x0 = np.array([[theta], [vtheta], [0.0]])
    s = System(W, b, bigX, ref, path)

    # Check if the initial state is inside the ellipsoid
    if (x0 - s.xstar).T @ P @ (x0 - s.xstar) <= 1.0:
      # Initial eta0 computation with respect to the initial state
      eta0 = ((1 - (x0 - s.xstar).T @ P @ (x0 - s.xstar)) / (s.nlayers * 2))[0][0]
      
      # Flag variable update to stop the search
      in_ellip = True
      
      initial_configs.append((theta, vtheta, ref, eta0)) 

      print(f"Initial state: theta0 = {theta*180/np.pi:.2f} deg, vtheta0 = {vtheta:.2f} rad/s, constant disturbance = {ref*180/np.pi:.2f} deg")
      print(f"Initial eta0: {eta0:.2f}")

initial_configs = np.array(initial_configs).squeeze()
# np.save('init_configs.npy', initial_configs)