import numpy as np

decay_rates = np.array([0.9, 0.9])
rhos = np.array([0.4, 0.3])
lambdas = decay_rates - rhos
gammas = (np.ones(2) - lambdas) / rhos
eta0 = 0.0