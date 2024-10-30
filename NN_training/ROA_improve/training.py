from NN_training.models.NeuralNet_simple import NeuralNet
import torch
import torch.optim as optim
import torch.nn as nn
from stable_baselines3 import PPO
from NN_training.environments import PendulumEnv

model = NeuralNet()
model.load_state_dict(torch.load('model.pth'))
model.eval()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if __name__ == "__main__":
  env = PendulumEnv(model)
  model_rl = PPO("MlpPolicy", env, verbose=1)
  model_rl.learn(total_timesteps=10000)
  torch.save(model.state_dict(), 'model_rl.pth')