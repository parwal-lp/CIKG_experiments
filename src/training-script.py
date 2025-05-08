from src.train import *
from src.solver import *

device = detectDevice()
[test_dataset, test_loader, posTests] = setTestDatasets()
[train_dataset, train_loader] = setTrainDatasets()

SLPmodels = trainModels('SLP', train_loader, test_loader, device)

MLPmodels = trainModels('MLP', train_loader, test_loader, device)