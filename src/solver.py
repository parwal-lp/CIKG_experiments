from z3 import *
import matplotlib.pyplot as plt
import torch
import numpy as np

def draw(arr):
  tens = torch.tensor(arr)
  img = np.array(tens, dtype=np.uint8).reshape((28, 28))
  img = np.array(img).reshape((28, 28))
  plt.imshow(img, cmap='gray', vmin=0, vmax=255)
  plt.axis('off')
  plt.title(f"esempio positivo")
  plt.show()
  print(arr)

def isValid(witness, q, x_vars, device = 'cuda'):
    flag = True
    witness_values = [witness[x_var].as_long() for x_var in x_vars]
    draw(witness_values)
    for model in q:
        out = model(torch.FloatTensor(witness_values).to(device))
        print("il classificatore dice:", out)
        prediction = torch.heaviside(out, torch.tensor([1.0]).to(device))
        if prediction==0:
            print("Non è istanza positiva del modello")
            flag = False
    if flag==True:
        print("Istanza positiva di q!")

def checkSLP(q):
  s = Solver()

  x_vars = [Int(f'x_{i}') for i in range(784)]# crea variabili (una per ogni pixel: 28*28=784)
  for x, i in zip(x_vars, range(len(x_vars))):
    s.add(x >= 0)
    s.add(x <= 255)
  #s.add(Sum([x_vars[i] for i in range(len(x_vars))]) > 50000)
  for model, i in zip(q, range(len(q))):
    W = list(model.parameters())[0].data
    b = list(model.parameters())[1].data
    W = torch.flatten(W)
    b = b.item()

    expr = Sum([RealVal(W[i].item()) * x_vars[i] for i in range(len(W))]) # combinazione lineare variabili e pesi in W
    #print(expr.sexpr())
    s.add(expr + b >= 0) # formula finale (W*x + b >= 0)

  # verifico se è satisfiable, ovvero se esiste una soluzione al sistema di disequazioni
  res = s.check()
  print(res)
  if res == sat:
      witness = s.model()
      isValid(witness, q, x_vars)
  
def checkMLP(q, h_size):
  s = Solver()
  in_size = 28*28

  x_vars = [Int(f'x_{i}') for i in range(in_size)]# crea variabili (una per ogni pixel: 28*28=784)
  #s.add(Sum([ x_vars[i] for i in range(in_size)]) >= 20000)
  for x in x_vars:
    s.add(x >= 0)
    s.add(x <= 255)

  for model in q:
    W_1 = list(model.parameters())[0].data
    W_1 = torch.transpose(W_1, 0, 1)
    b_1 = list(model.parameters())[1].data
    W_2 = list(model.parameters())[2].data
    b_2 = list(model.parameters())[3].data
    W_2 = torch.flatten(W_2)
    b_2 = b_2.item()


    # Somma del primo layer (combinazione lineare)
    y1 = []
    print("codifico primo layer")
    for j in range(h_size):
      weighted_sum = Sum([ x_vars[i] * RealVal(W_1[i][j].item()) for i in range(in_size)])
      y1_j = weighted_sum + RealVal(b_1[j].item())
      y1.append(y1_j)
    print("codifico secondo layer")
    y2 = [If(y1[j]>0, y1[j], 0) for j in range(h_size)] #relu effettiva
    #y2 = y1 #salto layer non lineare
    print("codifico terzo layer")
    y3 = Sum([RealVal(W_2[j].item()) * y2[j] for j in range(h_size)]) + RealVal(b_2)

    s.add(y3 >= 0)

    print("lancio il solver")
    res = s.check()
    print(res)
    if res == sat:
        print("cerco il witness")
        witness = s.model()
        witness_values = [witness[x_var].as_long() for x_var in x_vars]
        isValid(witness_values)
  
