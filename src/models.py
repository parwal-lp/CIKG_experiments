import torch.nn as nn
import torch

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
    
class FlexMLP(nn.Module):
    def __init__(self, in_size, h_size, out_size):
        super().__init__()
        self.in_size = in_size
        self.h_size = h_size
        self.fc1 = nn.Linear(in_size, h_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(h_size, out_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(out_size, 1, bias=False)

    def forward(self, x):
        x = x.view(-1, self.in_size)
        y1 = self.fc1(x)
        y2 = self.relu1(y1)
        y3 = self.fc2(y2)
        y4 = self.relu2(y3)
        y5 = self.fc3(y4)
        return y5
'''
# questa sarebbe l'implementazione esatta della rete combinata, ma Marabou non riesce a risolverla 
# per via di heaiside e prod che non sono supportate neanche nel formato onnx sembrerebbe
    def forward(self, x):
        x = x.view(-1, self.in_size)
        y1 = self.fc1(x)
        y2 = self.relu(y1)
        y3 = self.fc2(y2)
        y4 = torch.heaviside(y3, torch.tensor([1.0]))
        out = torch.prod(y4)
        return out
'''