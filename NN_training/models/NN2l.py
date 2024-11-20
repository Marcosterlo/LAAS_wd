import torch.nn as nn

class NeuralNet(nn.Module):
  def __init__(self, input_size):
    super(NeuralNet, self).__init__()
    self.l1 = nn.Linear(input_size, 16)
    self.hardtanh1 = nn.Hardtanh()
    self.l2 = nn.Linear(16, 16)
    self.hardtanh2 = nn.Hardtanh()
    self.l3 = nn.Linear(16, 1)
  
  def forward(self, x):
    x = self.hardtanh1(self.l1(x))
    x = self.hardtanh2(self.l2(x))
    x = self.l3(x)
    return x