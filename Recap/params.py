import numpy as np

# decay_rates = np.array([0.9, 0.9, 0.9, 0.9])
# rhos = np.array([0.6, 0.4, 0.2, 0.2])
# lambdas = decay_rates - rhos

rhos = np.ones(4) * np.load('finsler/Rho.npy')[0][0]
# gamma = 1.0 # works
gamma = 0.8
eta0 = 0.0