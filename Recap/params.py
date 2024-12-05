import numpy as np

# Vector of rhos imported from LMI solution
rhos = np.ones(4) * np.load('finsler/Rho.npy')[0][0]

# Gamma values found for bisection

# Minimum gamma for lambda = 0
# gamma = 0.6 # works for every lambda
gamma = 0.5

# Maximum gamma for lambda = 1
gamma = 100 

# When the 2 values will be found gamma will be dealt with as:
# gamma = lower_gamma +  lamda * (upper_gamma - lower_gamma)

# Old gamma value for old solution
gamma = 2