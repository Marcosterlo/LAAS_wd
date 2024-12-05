from LMI import LMI
import os
import numpy as np

# Weights and bias import
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

path = 'parameters_search'

# Lmi object creation
lmi = LMI(W, b)

# Alpha search 
alpha = lmi.search_alpha(1.0, 0.0, 1e-2, 1.0, 1.0, verbose=True)
np.save(path + '/alpha.npy', alpha)

# High gamma serach
high_gamma = lmi.search_highest_gamma(2.0, 0.0, 1e-2, alpha, verbose=True)
np.save(path + '/high_gamma.npy', high_gamma)

# Low gamma search
low_gamma = lmi.search_lowest_gamma(2.0, 0.0, 1e-2, alpha, verbose=True)
np.save(path + '/low_gamma.npy', low_gamma)

# Vector of lambda values
lambdas = np.linspace(0, 1, 50)

i = 0
for lam in lambdas:
  lmi.solve(alpha, lam, None, verbose=True, search=False)
  lmi.save_results(path + f'/lambda_{i}')
  i += 1