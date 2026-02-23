#codifico la query Zero and One in una unica neural network
import numpy as np
from src.solver import *
from src.train import *

def getModels(modelType):
    device = detectDevice()
    models = loadModels(modelType, device)
    return models

models = getModels(modelType='MLP')

model0 = models[0]
model1 = models[1]

zeroArray = np.zeros([1, 16])

W0_1 = list(model0.parameters())[0]
b0_1 = list(model0.parameters())[1]
W0_2 = list(model0.parameters())[2]
b0_2 = list(model0.parameters())[3]

W1_1 = list(model1.parameters())[0]
b1_1 = list(model1.parameters())[1]
W1_2 = list(model1.parameters())[2]
b1_2 = list(model1.parameters())[3]



Wcomb_1 = torch.tensor(np.vstack((W0_1.detach().cpu().numpy(), W1_1.detach().cpu().numpy())))

bcomb_1 = torch.tensor(np.hstack((b0_1.detach().cpu().numpy(), b1_1.detach().cpu().numpy())))

Wcomb_2_first_row = np.hstack((W0_2.detach().cpu().numpy(), zeroArray))
Wcomb_2_second_row = np.hstack((zeroArray, W1_2.detach().cpu().numpy()))
Wcomb_2 = torch.tensor(np.vstack((Wcomb_2_first_row, Wcomb_2_second_row)))

bcomb_2 = torch.tensor(np.hstack((b0_2.detach().cpu().numpy(), b1_2.detach().cpu().numpy())))

input_size = 28*28
h_size = 32
out_size = 2

modelComb = FlexMLP(input_size, h_size, out_size)
with torch.no_grad():
    modelComb.fc1.weight.copy_(Wcomb_1)
    modelComb.fc1.bias.copy_(bcomb_1)
    modelComb.fc2.weight.copy_(Wcomb_2)
    modelComb.fc2.bias.copy_(bcomb_2)
    modelComb.fc3.weight.copy_(torch.tensor([-1.0, -1.0]))

print("-----------------")
out = modelComb(torch.zeros(784))
print(out)
print("-----------------")

onnx_program = torch.onnx.export(modelComb, torch.zeros(784), "codifica_0and1.onnx")