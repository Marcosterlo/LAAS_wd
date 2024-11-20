import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from NN_training.models.NN2l import NeuralNet

df = pd.read_csv('dataset.csv')
states = df[['theta', 'vtheta', 'eta']].values
inputs = df[['u']].values

states_tensor = torch.tensor(states, dtype=torch.float32)
input_tensor = torch.tensor(inputs, dtype=torch.float32)

dataset = TensorDataset(states_tensor, input_tensor)

batch_size = 1024
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = NeuralNet(3).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
  for states, inputs in dataloader:
    states, inputs = states.to(device), inputs.to(device)
    outputs = model(states)
    loss = criterion(outputs, inputs)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
  print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
  
  if device.type == 'cuda':
    allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved_memory = torch.cuda.memory_reserved(device) / (1024 ** 2)
    print(f"GPU memory allocated: {allocated_memory:.2f} MB")
    print(f"GPU memory cached:    {reserved_memory:.2f} MB")

torch.save(model.state_dict(), 'model2l.pth')