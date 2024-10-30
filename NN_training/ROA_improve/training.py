from NN_training.models.NeuralNet_simple import NeuralNet
import torch
import torch.optim as optim
import torch.nn as nn

model = NeuralNet()
model.load_state_dict(torch.load('model.pth'))
model.eval()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if __name__ == "__main__":
  pass