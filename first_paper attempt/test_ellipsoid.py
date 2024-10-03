import numpy as np
import matplotlib.pyplot as plt

P1_vals = []
P2_vals = []
P01_vals = []
P02_vals = []

P = np.array([[0.20944529, 0.02045225], [0.02045225, 0.01260998]])
P0 = np.array([[0.3024, 0.0122], [0.0122, 0.0154]])

for i in range(100000):
    x1 = np.random.uniform(-10, 10)
    x2 = np.random.uniform(-10, 10)
    vec = np.array([x1, x2])
    if (vec.T @ P @ vec < 1):
        P1_vals.append(x1)
        P2_vals.append(x2)
    if (vec.T @ P0 @ vec < 1):
        P01_vals.append(x1)
        P02_vals.append(x2)

plt.plot(P1_vals, P2_vals)
plt.plot(P01_vals, P02_vals)
plt.grid(True)
plt.show()