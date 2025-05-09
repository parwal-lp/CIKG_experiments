import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.utils.data import Subset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, accuracy_score
import matplotlib.pyplot as plt
from src.models import *

def detectDevice():
    print('Using PyTorch version:', torch.__version__)
    if torch.cuda.is_available():
        print('Using GPU, device name:', torch.cuda.get_device_name(0))
        device = torch.device('cuda')
    else:
        print('No GPU found, using CPU instead.')
        device = torch.device('cpu')
    return device

def setTrainDatasets():
    batch_size = 32
    data_dir = './data'

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=ToTensor())
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    return [train_dataset, train_loader]

def setTestDatasets():
    batch_size = 32
    data_dir = './data'

    test_dataset = datasets.MNIST(data_dir, train=False, transform=ToTensor())
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    posTests = []
    for i in range(10):
      indices = [j for j, (img, label) in enumerate(test_dataset) if label == i]
      ds = Subset(test_dataset, indices)
      posTests.append(DataLoader(dataset=ds, batch_size=batch_size, shuffle=False))

    return [test_dataset, test_loader, posTests]


def train(data_loader, model, criterion, optimizer, classNumber, device):
    model.train()

    num_batches = len(data_loader)

    total_loss = 0

    all_targets = []
    all_outputs = []

    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        target = (target == classNumber).type(torch.float).reshape(-1, 1)
        
        output = model(data)
        binary_output = (output >= 0).type(torch.float).reshape(-1, 1)

        loss = criterion(output, target)
        total_loss += loss

        all_targets.extend(target.cpu().numpy())
        all_outputs.extend(binary_output.cpu().numpy())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_loss = total_loss/num_batches
    accuracy = accuracy_score(all_targets, all_outputs)
    recall = recall_score(all_targets, all_outputs)
    print(f"Average loss: {train_loss:7f}, accuracy: {accuracy:.2%}, recall: {recall}")

def test(test_loader, model, classNumber, device):
    model.eval()

    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            target = (target == classNumber).type(torch.float).reshape(-1, 1)

            # Do a forward pass
            output = model(data)
            output = (output >= 0).type(torch.float).reshape(-1, 1)

            # Count number of correct digits
            all_targets.extend(target.cpu().numpy())
            all_outputs.extend(output.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_outputs)
    recall = recall_score(all_targets, all_outputs)

    print(f"accuracy: {accuracy}, recall: {recall}")
    return [accuracy, recall]


def plot_confusion_matrix(model, data_loader, classNumber, device):
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            binary_labels = (labels == classNumber).int()
            outputs = model(images).view(-1)
            preds = (outputs >= 0).int()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(binary_labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["non-"+str(classNumber), str(classNumber)])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f"Confusion Matrix for class {classNumber}")
    plt.show()


def trainModels(modelType, train_loader, device):
    models = []
    criterions = []
    optimizers = []

    pos_weight = torch.tensor([9.0]).to(device) #peso ai sample positivi per contrastare sbilanciamento dataset (istanze positive sono solo il 10% del dataset)

    for i in range(10):
      if modelType=='SLP':
          model = SimpleSLP().to(device)
      elif modelType=='MLP':
          model = SimpleMLP().to(device)
      else:
          print("unknown model required")
      models.append(model)
      criterions.append(nn.BCEWithLogitsLoss(pos_weight=pos_weight))
      optimizers.append(torch.optim.Adam(models[i].parameters()))
    print(models)
    print(criterions)
    print(optimizers)

    epochs = 10
    for i in range(0, 10):
      print(f"----------------------Training {modelType} classifier for {i}----------------------------")
      for epoch in range(epochs):
          print(f"Training epoch: {epoch+1}")
          train(train_loader, models[i], criterions[i], optimizers[i], i, device)

    for i in range(len(models)):
        torch.save(models[i].state_dict(), f'./models/{modelType}/model_{i}')

    return models

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
        models[i].load_state_dict(torch.load(f'./models/{modelType}/model_{i}', weights_only=True))
        print(f"loaded classifier model for {i}")
    return models

def testModels(models, test_loader, device):
    tot_acc = 0
    tot_rec = 0
    for i in range(len(models)):
      print(f"summary classifier model for {i}")
      acc, rec = test(test_loader, models[i], i, device)
      plot_confusion_matrix(models[i], test_loader, i, device)
      tot_acc += acc
      tot_rec += rec
    print(f"Average accuracy: {tot_acc/len(models)}, Average recall: {tot_rec/len(models)}")