import numpy as np

decay_rates = np.array([0.9, 0.9])
rhos = np.array([0.4, 0.3])
lambdas = decay_rates - rhos
gammas = (lambdas - np.ones(2)) / rhos
eta0 = 0.0