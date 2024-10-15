import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

df = pd.read_csv('dataset.csv')
states = df[['theta', 'vtheta', 'eta']].values
inputs = df[['u']].values

states_tensor = torch.tensor(states, dtype=torch.float32)
input_tensor = torch.tensor(inputs, dtype=torch.float32)

dataset = TensorDataset(states_tensor, input_tensor)

batch_size = 1024
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class NeuralNet(nn.Module):
  def __init__(self):
    super(NeuralNet, self).__init__()
    self.l1 = nn.Linear(3, 32)
    self.hardtanh1 = nn.Hardtanh()
    self.l2 = nn.Linear(32, 32)
    self.hardtanh2 = nn.Hardtanh()
    self.l3 = nn.Linear(32, 32)
    self.hardtanh3 = nn.Hardtanh()
    self.l4 = nn.Linear(32, 1)
  
  def forward(self, x):
    x = self.hardtanh1(self.l1(x))
    x = self.hardtanh2(self.l2(x))
    x = self.hardtanh3(self.l3(x))
    x = self.l4(x)
    return x

model = NeuralNet()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if __name__ == "__main__":
  num_epochs = 100
  for epoch in range(num_epochs):
    for states, inputs in dataloader:
      outputs = model(states)
      loss = criterion(outputs, inputs)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

  torch.save(model.state_dict(), 'test.pth')