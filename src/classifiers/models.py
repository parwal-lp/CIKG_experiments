import torch.nn as nn
import torch

in_size = 28*28
h_size = 16
out_size = 1

class SimpleSLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(in_size, 1)

    def forward(self, x):
        x = x.view(-1, in_size)
        x = self.fc(x)
        return x


class SimpleMLP(nn.Module):
    def __init__(self, in_size=in_size, h_size=h_size, out_size=out_size):
        super().__init__()
        self.fc1 = nn.Linear(in_size, h_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(h_size, out_size)

    def forward(self, x):
        x = x.view(-1, in_size)
        y1 = self.fc1(x)
        y2 = self.relu(y1)
        y3 = self.fc2(y2)
        return y3
    
class OntologyAndQueryNetwork(nn.Module):
    def __init__(self, n_classifiers, n_disj, n_axioms, out_size=2):
        super().__init__()
        self.fc1 = nn.Linear(n_classifiers, n_classifiers, bias=False)
        self.fc2 = nn.Linear(n_classifiers, n_classifiers)
        self.fc3 = nn.Linear(n_classifiers, n_disj+n_axioms)
        self.fc4 = nn.Linear(n_disj+n_axioms, out_size)

    def forward(self, x):
        x = x.view(-1, in_size)
        y1 = self.fc1(x)
        y2 = torch.sign(y1)
        y3 = self.fc2(y2)
        y4 = torch.sign(y3)
        y5 = self.fc3(y4)
        y6 = torch.sign(y5)
        out = self.fc4(y6)
        return out
    
class finalEncodedNetwork(nn.Module):
    def __init__(self, mergedClassifiers, encodedQueryOntology):
        super().__init__()
        self.mergingLayer = mergedClassifiers
        self.encodingLayer = encodedQueryOntology
    def forward(self, x):
        y1 = self.mergingLayer(x)
        out = self.encodingLayer(y1)
        return out
    
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