import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class System:
  
  def __init__(self):
    
    ## Parameters
    self.g = 9.81 # Gravity coefficient
    self.m = 0.15 # mass
    self.l = 0.5 # length
    self.mu = 0.05 # frict coeff
    self.dt = 0.02 # sampling period
    self.reference = 0

    ## System definition x^+ = A*x + B*u

    self.A = np.array([[1,      self.dt, 0],
                  [self.g/self.l*self.dt, 1 - self.mu/(self.m*self.l**2)*self.dt, 0],
                  [1, 0, 1]])

    self.B = np.array([[0],
                  [self.dt/(self.m*self.l**2)],
                  [0]])

    # Gain matrix
    self.K = np.array([11.04825000000001, 1.075000000000001, 0.562500000000001]).reshape(1, 3)

    self.state = np.array([[0.1], [1], [0]])

  def step(self):
    u = -self.K @ self.state
    newx = self.A @ self.state + self.B * u
    newx[2] += -self.reference
    self.state = newx
    return self.state, u

if __name__ == "__main__":

  s = System()

  n_samples = 10000
  n_steps = 100

  theta_lim = 60*np.pi/180
  vtheta_lim = 5

  dataset = []

  for i in range(n_samples):
    theta = np.random.uniform(-theta_lim, theta_lim)
    vtheta = np.random.uniform(-vtheta_lim, vtheta_lim)
    s.state = np.array([[theta], [vtheta], [0]])
    for i in range(n_steps):
      state, u = s.step()
      data = np.concatenate([state, u], axis=0)
      dataset.append(data)
  
  dataset = np.array(dataset).reshape(n_samples*n_steps, 4)
  np.random.shuffle(dataset)

  column_labels = ['theta', 'vtheta', 'eta', 'u']
  df = pd.DataFrame(dataset, columns=column_labels) 
  df.to_csv('dataset.csv', index=False)