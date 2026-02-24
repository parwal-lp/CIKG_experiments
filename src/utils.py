import torch
from src.classifiers.models import *
from src.classifiers.train import *

def detectDevice():
    print('Using PyTorch version:', torch.__version__)
    if torch.cuda.is_available():
        print('Using GPU, device name:', torch.cuda.get_device_name(0))
        device = torch.device('cuda')
    else:
        print('No GPU found, using CPU instead.')
        device = torch.device('cpu')
    return device


def loadModels(modelType, device):
    models = []
    for i in range(10):
      if modelType=='SLP':
          model = SimpleSLP().to(device)
      elif modelType=='MLP':
          model = SimpleMLP().to(device)
      else:
          print("unknown model required")
      models.append(model)
    for i in range(10):
        models[i].load_state_dict(torch.load(f'./models/{modelType}/model_{i}', weights_only=True, map_location=device))
        print(f"loaded classifier model for {i}")
    return models

def getModels(modelType, testFlag=False, even=False):
    device = detectDevice()

    models = loadModels(modelType, device)
    if testFlag:
        testModels(models, test_loader, device)

    if even:
        SLPeven = SimpleSLP().to(device)
        SLPeven.load_state_dict(torch.load(f'./models/SLP/model_[0, 2, 4, 6, 8]', weights_only=True, map_location=device))
        if(testFlag): 
            [test_dataset, test_loader, posTests] = setTestDatasets()
            test(test_loader, SLPeven, [2,4,6,8], device)
        return models, SLPeven

    return models
