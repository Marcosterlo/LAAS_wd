import numpy as np

class Integrator():

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
        self.nx = 3
        self.nu = 1
    
    def step(self, input):

        x, dx, eta = self.state
        g = self.g
        m = self.m
        l = self.l
        mu = self.mu
        dt = self.dt

        y = x - self.constant_reference

        u = np.squeeze(np.clip(input, -1, 1) * self.max_torque)

        dxplus = dx + (g/l*np.sin(x) - mu/(m*l**2)*dx + 1/(m*l**2)*u)*dt
        xplus = x + dx * dt
        etaplus = eta + y
        
        self.state = np.array([xplus, dxplus, etaplus])

        return self.state