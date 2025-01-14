from NN_training.models.NN2l import NeuralNet
import torch
import pandas as pd

model = NeuralNet(3)
model.load_state_dict(torch.load('model2l.pth'))

weight_and_biases = {}

for name, param in model.named_parameters():
  if 'weight' in name:
    weight_and_biases[name] = param.detach().numpy()
  elif 'bias' in name:
    weight_and_biases[name] = param.detach().numpy()

for name, value in weight_and_biases.items():
  df = pd.DataFrame(value)
  filename = f'./weights2l/{name}.csv'
  df.to_csv(filename, index=False, header=False)