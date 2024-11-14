import numpy as np

decay_rates = np.array([0.9, 0.9, 0.9])
rhos = np.array([0.4, 0.3, 0.2])
lambdas = decay_rates - rhos
gammas = (np.ones(3) - lambdas) / rhos
eta0 = 10