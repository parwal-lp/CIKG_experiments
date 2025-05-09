from src.train import *
from src.solver import *

#detect available device: CPU or GPU
device = detectDevice()

#load datasets
[test_dataset, test_loader, posTests] = setTestDatasets()
[train_dataset, train_loader] = setTrainDatasets()

#train all models
SLPmodels = trainModels('SLP', train_loader, device)
MLPmodels = trainModels('MLP', train_loader, device)

#test all models
testModels(SLPmodels, test_loader, device)
testModels(MLPmodels, test_loader, device)