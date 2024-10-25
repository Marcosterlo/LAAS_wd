import numpy as np

decay_rates = np.array([0.9, 0.9, 0.9])
rhos = np.array([0.4, 0.3, 0.2])
lambdas = decay_rates - rhos
gammas = (np.ones(3) - lambdas) / rhos
eta0 = 10

# Eta0 = 100 is a good value for which etas remain positive. This means that in one step the system is able to come back inside the sector conditons of the activation functions. 
