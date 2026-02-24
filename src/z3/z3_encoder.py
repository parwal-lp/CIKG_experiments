import time
from z3 import *
import torch
import os
from datetime import datetime

def saveProgram(s, file_path):
  try:
    dir_name = os.path.dirname(file_path)
    if dir_name:
      os.makedirs(dir_name, exist_ok=True)
    if hasattr(s, 'to_smt2'):
      content = s.to_smt2()
    else:
      content = str(s)
    with open(file_path, 'w') as f:
      f.write(content if isinstance(content, str) else str(content))
    print(f"Program dumped to {file_path}")
  except Exception as e:
    print(f"Failed to dump program: {e}")


def generateSimpleProgram(q, mode, in_size=10, h_size=6, dom="integer", tactics="no"):
  if tactics=="yes":
    print("using tactics")
    s = Goal()
  else:
    s = Solver()

  if dom=="integer":
    x_vars = [Int(f'x_{i}') for i in range(in_size)] #create variables (one per each input feature)

  elif dom=="real":
    x_vars = [Real(f'x_{i}') for i in range(in_size)] #create variables (one per each input feature)

  for x in x_vars:
    s.add(x >= 0)
    s.add(x <= 1)

  for model in q:
    W_1 = list(model.parameters())[0].data
    W_1 = torch.transpose(W_1, 0, 1)
    b_1 = list(model.parameters())[1].data
    W_2 = list(model.parameters())[2].data
    b_2 = list(model.parameters())[3].data
    W_2 = torch.flatten(W_2)
    b_2 = b_2.item()


    # linear combination - first layer
    y1 = []
    print("encode first layer")
    for j in range(h_size):
      weighted_sum = Sum([ x_vars[i] * W_1[i][j].item() for i in range(in_size)])
      y1_j = weighted_sum + b_1[j].item()
      y1.append(y1_j)

    # ReLU - second layer
    print("encode second layer")
    y2 = [If(y1[j]>0, y1[j], 0) for j in range(h_size)] #relu encoding

    # linear combination - third layer
    print("encode third layer")
    y3 = Sum([W_2[j].item() * y2[j] for j in range(h_size)]) + b_2

    # add inequality to the system
    s.add(y3 >= 0)

  print("SMT encoding completed")
  if (mode == 'print'):
    try:
      project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
      saveProgram(s, os.path.join(project_root, 'logs', 'simple.smt2'))
    except Exception as e:
      print(f"Could not write solver to file: {e}")
    return
  
  elif (mode == 'solve'):
    if tactics=="yes":
      t1 = Tactic('simplify')
      t2 = Tactic('solve-eqs')
      t  = Then(t1, t2)
      s = t(s)

    print("start solver")
    start_time = time.time()
    res = s.check()
    end_time = time.time()
    #riporto tempistiche del solver
    elapsed = end_time - start_time
    print(f"Execution time: {elapsed:.4f}s")
    print(res)
    if res == sat:
      print("Model (witness):")
      witness = s.model()
      print(witness)
    return

def generateProgram(q, modelType='MLP'):
  in_size = 28*28
  h_size = 50
  s = Solver()

  x_vars = [Int(f'x_{i}') for i in range(in_size)] #create variables (one per each input feature)

  for x in x_vars:
    s.add(x >= 0)
    s.add(x <= 1)

  #CASO SLP
  if modelType=='SLP':
    for model, i in zip(q, range(len(q))):
      W = list(model.parameters())[0].data
      b = list(model.parameters())[1].data
      W = torch.flatten(W)
      b = b.item()

      expr = Sum([RealVal(W[i].item()) * x_vars[i] for i in range(len(W))]) # linear combination - perceptron layer
      s.add(expr + b >= 0) #final inequality for the model (W*x + b >= 0)

  #CASO MLP
  if modelType=='MLP':
    for model in q:
      W_1 = list(model.parameters())[0].data
      W_1 = torch.transpose(W_1, 0, 1)
      b_1 = list(model.parameters())[1].data
      W_2 = list(model.parameters())[2].data
      b_2 = list(model.parameters())[3].data
      W_2 = torch.flatten(W_2)
      b_2 = b_2.item()


      # linear combination - first layer
      y1 = []
      print("encode first layer")
      for j in range(h_size):
        weighted_sum = Sum([ x_vars[i] * W_1[i][j].item() for i in range(in_size)])
        y1_j = weighted_sum + b_1[j].item()
        y1.append(y1_j)

      # ReLU - second layer
      print("encode second layer")
      # y2 = [If(y1[j]>0, y1[j], 0) for j in range(h_size)] #relu encoding
      y2 = [y1[j]*2 for j in range(h_size)] #linear encoding
      # y2 = [y1[j]*y1[j] for j in range(h_size)] #quadratic encoding

      # linear combination - third layer
      print("encode third layer")
      y3 = Sum([W_2[j].item() * y2[j] for j in range(h_size)]) + b_2

      # add inequality to the system
      s.add(y3 >= 0)

  print("SMT encoding completed")

  try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    programs_dir = os.path.join(project_root, 'generated_programs')
    os.makedirs(programs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"mnist_{timestamp}.smt2"
    filepath = os.path.join(programs_dir, filename)
    saveProgram(s, filepath)
    print(f"SMT program written to generated_programs/{filename}")
  except Exception as e:
    print(f"Could not write solver to file: {e}")  
  return