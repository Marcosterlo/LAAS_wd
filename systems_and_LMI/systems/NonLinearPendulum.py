from systems_and_LMI.systems.LinearPendulum_integrator import LinPendulumIntegrator
import numpy as np
import os

# New class definition that includes non linear behavior
class NonLinPendulum(LinPendulumIntegrator):
  
  # Calling the constructor of the parent class
  def __init__(self):
    super().__init__()

    # Addition of constant reference to test robustness
    self.constant_reference = 0.0

    # Matrix to distribute non-linearity to the state
    self.C = np.array([
      [0],
      [self.g / self.l * self.dt],
      [0]
    ])

    # Get the directory of the current file
    current_dir = os.path.dirname(__file__)
    # Construct the full path to the K.npy file
    k_file_path = os.path.join(current_dir, 'K.npy')
    # Load the matrix K from the K.npy file and invert the sign to have the input shaped like u = Kx
    self.K = -np.load(k_file_path)

  def step(self, input):

    # Non linear term in the form sin(theta) - theta
    nonlin = np.sin(self.state[0]) - self.state[0]
    
    # State computation
    self.state = self.A @ self.state + self.B @ input + self.C * nonlin
    
    # Addition of constant reference to integrator term of the state
    self.state[2] += -self.constant_reference
    
    # returns the updated state
    return self.state

if __name__ == "__main__":
  s = NonLinPendulum()