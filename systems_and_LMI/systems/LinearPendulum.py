import numpy as np

class LinPendulum():
  def __init__(self):
    self.state = None
    self.g = 9.81
    self.m = 0.15
    self.l = 0.5
    self.mu = 0.05
    self.dt = 0.02
    self.max_torque = 5
    self.max_speed = 8.0
    self.constant_reference = 0
    self.nx = 2
    self.nu = 1

    self.A = np.array([
        [1, self.dt],
        [self.g*self.dt/self.l, 1-self.mu*self.dt/(self.m*self.l**2)]
    ])

    self.B = np.array([
        [0],
        [self.dt/(self.m*self.l**2)]
    ])

  def step(self, input):
    self.state = self.A @ self.state + self.B @ input
    return self.state

if __name__ == "__main__":
  s = LinPendulum()