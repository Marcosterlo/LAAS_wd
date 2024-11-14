from NN_training.models.NeuralNet_simple import NeuralNet
import torch
from stable_baselines3 import PPO
import pandas as pd

model = PPO.load('best_so_far_3.zip')

weight_and_biases = {}

for name, param in model.policy.named_parameters():
  if 'policy' in name or 'action' in name:
    if 'weight' in name:
      weight_and_biases[name] = param.detach().numpy()
    elif 'bias' in name:
      weight_and_biases[name] = param.detach().numpy()
      
for name, value in weight_and_biases.items():
  df = pd.DataFrame(value)
  filename = f'./new_weights/{name}.csv'
  df.to_csv(filename, index=False, header=False)  