import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.utils.data import Subset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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

    #print(torch.__version__)
    #print(torch.version.cuda)
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

def correct(output, target, device='cuda'):
    predicted_digits = torch.heaviside(output, torch.tensor([1.0]).to(device)) #se output>=0 allora ritorna 1, altrimenti 0
    correct_ones = (predicted_digits == target).type(torch.float) #vettore con 1 dove la predizione era corretta, 0 dove era sbagliata
    return correct_ones.sum().item() #conta i corretti

def train(data_loader, model, criterion, optimizer, classNumber, device='cuda'):
    model.train()

    num_batches = len(data_loader)
    num_items = len(data_loader.dataset)

    total_loss = 0
    total_correct = 0
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        # converte le label in 1 per il valore scelto e 0 per tutti gli altri valori
        target = (target == classNumber).type(torch.float).reshape(-1, 1)
        #print("converted label:", target.flatten())
        

        output = model(data)
        #print("prediction: ", output.flatten())
        loss = criterion(output, target)
        total_loss += loss

        total_correct += correct(output, target)
        #print("correct: ", correct(output, target))
        #print("correct: ", total_correct)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_loss = total_loss/num_batches
    accuracy = total_correct/num_items
    print(f"Average loss: {train_loss:7f}, accuracy: {accuracy:.2%}")

def test(test_loader, model, classNumber, device='cuda'):
    model.eval()

    num_batches = len(test_loader)
    num_items = len(test_loader.dataset)

    test_loss = 0
    total_correct = 0

    pos_weight = torch.tensor([9.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    with torch.no_grad():
        for data, target in test_loader:
            # Copy data and targets to GPU
            data = data.to(device)
            target = target.to(device)
            target = (target == classNumber).type(torch.float).reshape(-1, 1)

            # Do a forward pass
            output = model(data)

            # Calculate the loss
            loss = criterion(output, target)
            test_loss += loss.item()

            # Count number of correct digits
            total_correct += correct(output, target)

    test_loss = test_loss/num_batches
    accuracy = total_correct/num_items

    print(f"Testset accuracy: {100*accuracy:>0.1f}%, average loss: {test_loss:>7f}")


def plot_confusion_matrix(model, data_loader, classNumber, device='cuda'):
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


def trainModels(modelType, train_loader, test_loader, device='cuda'):
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
          train(train_loader, models[i], criterions[i], optimizers[i], i)
      #test(test_loader, models[i], criterions[i], i)
      #plot_confusion_matrix(models[i], test_loader, i)

    for i in range(len(models)):
        torch.save(models[i].state_dict(), f'./models/{modelType}/model_{i}')

    return models

def loadModels(modelType, device='cuda'):
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

def testModels(models, test_loader):
    for i in range(len(models)):
      print(f"summary classifier model for {i}")
      test(test_loader, models[i], i)
      #plot_confusion_matrix(models[i], test_loader, i)