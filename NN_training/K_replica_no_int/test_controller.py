from NN_training.models.NeuralNet_simple import NeuralNet
import torch
from NN_training.environments.simple_pendulum_env import Simple_pendulum_env
import numpy as np
import matplotlib.pyplot as plt

model = NeuralNet()
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu'), weights_only=True))

env = Simple_pendulum_env()

states = []
inputs = []

obs = env.reset()[0]
for i in range(1000):
  action = model.forward(torch.tensor(obs)).detach().numpy()
  obs, rewards, done, truncated, info = env.step(action)
  states.append(obs)
  inputs.append(action)

states = np.squeeze(np.array(states))
inputs = np.squeeze(np.array(inputs))*env.max_torque

plt.plot(states[:,0], states[:,1])
plt.show()

plt.plot(inputs)
plt.show()
