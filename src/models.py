import torch.nn as nn

in_size = 28*28

class SimpleSLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(in_size, 1)

    def forward(self, x):
        x = x.view(-1, in_size)
        x = self.fc(x)
        return x

h_size = 16

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_size, h_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(h_size, 1)

    def forward(self, x):
        x = x.view(-1, in_size)
        y1 = self.fc1(x)
        y2 = self.relu(y1)
        y3 = self.fc2(y2)
        return y3