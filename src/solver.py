import time
from z3 import *
import z3
import matplotlib.pyplot as plt
import torch
import numpy as np
from src.train import detectDevice
from dotenv import load_dotenv
import os

def saveSolver(s, file_path):
  try:
    dir_name = os.path.dirname(file_path)
    if dir_name:
      os.makedirs(dir_name, exist_ok=True)
    # Prefer SMT-LIB2 when available for full fidelity
    if hasattr(s, 'to_smt2'):
      content = s.to_smt2()
    else:
      content = str(s)
    with open(file_path, 'w') as f:
      f.write(content if isinstance(content, str) else str(content))
    print(f"Solver dumped to {file_path}")
  except Exception as e:
    print(f"Failed to dump solver: {e}")

def configureSolver(multithread):
  if multithread==True:
    load_dotenv()
    is_enabled = bool(os.getenv('Z3_PARALLEL', False))
    n_threads = int(os.getenv('Z3_THREADS', 1))
    z3.set_param('parallel.enable', is_enabled)
    z3.set_param('parallel.threads.max', n_threads)
    z3.set_param('smt.threads', n_threads)
    z3.set_param('sat.threads', n_threads)
    print(f"Z3 running on {n_threads} threads, parallel mode is {'enabled' if is_enabled else 'disabled'}.")
  else:
    print("Z3 running in single-threaded mode.")

def draw(arr):
  tens = torch.tensor(arr)
  img = np.array(tens, dtype=np.uint8).reshape((28, 28))
  img = np.array(img).reshape((28, 28))
  plt.imshow(img, cmap='gray', vmin=0, vmax=255)
  plt.axis('off')
  plt.title(f"esempio positivo")
  plt.show()
  print(arr)

def isValid(witness, q, x_vars):
    flag = True
    device = detectDevice()
    witness_values = [witness[x_var].as_long() for x_var in x_vars]
    draw(witness_values)
    for model in q:
        out = model(torch.FloatTensor(witness_values).to(device))
        print("classifier outputs:", out)
        prediction = torch.heaviside(out, torch.tensor([1.0]).to(device))
        if prediction==0:
            print("doesn't satisfy the classifier")
            flag = False
    if flag==True:
        print("Satisfies q!")
  
def isValidWithNeg(witness, qPos, qNeg, x_vars):
    flag = True
    device = detectDevice()
    witness_values = [witness[x_var].as_long() for x_var in x_vars]
    # draw(witness_values)
    for model in qPos:
        out = model(torch.FloatTensor(witness_values).to(device))
        print("classifier outputs:", out)
        prediction = torch.heaviside(out, torch.tensor([1.0]).to(device))
        if prediction==0:
            print("doesn't satisfy the classifier")
            flag = False
    for model in qNeg:
        out = model(torch.FloatTensor(witness_values).to(device))
        print("classifier outputs:", out)
        prediction = torch.heaviside(out, torch.tensor([1.0]).to(device))
        if prediction==1:
            print("satisfies the classifier")
            flag = False
    if flag==True:
        print("Satisfies q!")

def checkSLP(q, incT=[], disjT=[], needWitness=True):
  s = Solver()

  x_vars = [Int(f'x_{i}') for i in range(784)] #create variables (one per each input pixel: 28*28=784)
  for x, i in zip(x_vars, range(len(x_vars))):
    s.add(x >= 0)
    s.add(x <= 255)
  #s.add(Sum([x_vars[i] for i in range(len(x_vars))]) > 50) #set magnitude
  for model, i in zip(q, range(len(q))):
    W = list(model.parameters())[0].data
    b = list(model.parameters())[1].data
    W = torch.flatten(W)
    b = b.item()

    expr = Sum([RealVal(W[i].item()) * x_vars[i] for i in range(len(W))]) # linear combination - perceptron layer
    #print(expr.sexpr())
    s.add(expr + b >= 0) #final inequality for the model (W*x + b >= 0)

  # check satisfiability of the inequalities system
  for (t1, t2) in incT:
      W2 = list(t2.parameters())[0].data
      b2 = list(t2.parameters())[1].data
      W2 = torch.flatten(W)
      b2 = b2.item()
      expr2 = Sum([RealVal(W2[i].item()) * x_vars[i] for i in range(len(W2))]) #linear combination

      W1 = list(t1.parameters())[0].data
      b1 = list(t1.parameters())[1].data
      W1 = torch.flatten(W)
      b1 = b1.item()
      expr1 = Sum([RealVal(W1[i].item()) * x_vars[i] for i in range(len(W1))]) #linear combination

      s.add(Implies(expr1 + b1 >= 0, expr2 + b2 >= 0))

  for (t1, t2) in disjT:
      W2 = list(t2.parameters())[0].data
      b2 = list(t2.parameters())[1].data
      W2 = torch.flatten(W)
      b2 = b2.item()
      expr2 = Sum([RealVal(W2[i].item()) * x_vars[i] for i in range(len(W2))]) #linear combination

      W1 = list(t1.parameters())[0].data
      b1 = list(t1.parameters())[1].data
      W1 = torch.flatten(W)
      b1 = b1.item()
      expr1 = Sum([RealVal(W1[i].item()) * x_vars[i] for i in range(len(W1))]) #linear combination
      
      s.add(Implies(expr1 + b1 >= 0, expr2 + b2 < 0))

  #s now contains the SMT encoding of the problem instance
  res = s.check()
  print(res)
  if res == sat and needWitness == True:
      witness = s.model()
      #isValid(witness, q, x_vars)


def checkSLPwithNeg(qPos, qNeg=[], incT=[], disjT=[], needWitness=True, magnitude = 0, withTactics=False):
  if withTactics==True:
    print("using tactics")
    s = Goal()
  else:
    s = Solver()

  x_vars = [Int(f'x_{i}') for i in range(784)] #create variables (one per each input pixel: 28*28=784)
  for x in x_vars:
    s.add(x >= 0)
    s.add(x <= 255)
  if (magnitude>0): s.add(Sum([x_vars[i] for i in range(len(x_vars))]) > magnitude) #magnitude per forzare non-trivial solutions
  for model in qPos:
    W = list(model.parameters())[0].data
    b = list(model.parameters())[1].data
    W = torch.flatten(W)
    b = b.item()

    expr = Sum([RealVal(W[i].item()) * x_vars[i] for i in range(len(W))]) # linear combination - perceptron layer
    #print(expr.sexpr())
    s.add(expr + b >= 0) #final inequality for the model (W*x + b >= 0)

  for model in qNeg:
    W = list(model.parameters())[0].data
    b = list(model.parameters())[1].data
    W = torch.flatten(W)
    b = b.item()

    expr = Sum([RealVal(W[i].item()) * x_vars[i] for i in range(len(W))]) # linear combination - perceptron layer
    #print(expr.sexpr())
    s.add(expr + b < 0) #final inequality for the model (W*x + b >= 0)

  # check satisfiability of the inequalities system
  for (t1, t2) in incT:
      W2 = list(t2.parameters())[0].data
      b2 = list(t2.parameters())[1].data
      W2 = torch.flatten(W2)
      b2 = b2.item()
      expr2 = Sum([RealVal(W2[i].item()) * x_vars[i] for i in range(len(W2))]) # linear combination

      W1 = list(t1.parameters())[0].data
      b1 = list(t1.parameters())[1].data
      W1 = torch.flatten(W1)
      b1 = b1.item()
      expr1 = Sum([RealVal(W1[i].item()) * x_vars[i] for i in range(len(W1))]) # linear combination
      #s.add(Or(expr1 + b1 < 0, expr2 + b2 >= 0))
      s.add(Implies(expr1 + b1 >= 0, expr2 + b2 >= 0))

  for (t1, t2) in disjT:
      W2 = list(t2.parameters())[0].data
      b2 = list(t2.parameters())[1].data
      W2 = torch.flatten(W2)
      b2 = b2.item()
      expr2 = Sum([RealVal(W2[i].item()) * x_vars[i] for i in range(len(W2))]) # linear combination

      W1 = list(t1.parameters())[0].data
      b1 = list(t1.parameters())[1].data
      W1 = torch.flatten(W1)
      b1 = b1.item()
      expr1 = Sum([RealVal(W1[i].item()) * x_vars[i] for i in range(len(W1))]) # linear combination
      #s.add(Or(expr1 + b1 < 0, expr2 + b2 < 0))
      s.add(Implies(expr1 + b1 >= 0, expr2 + b2 < 0))

  if withTactics==True:
    t1 = Tactic('simplify')
    t2 = Tactic('solve-eqs')
    t  = Then(t1, t2)
    r = t(s)
    print("result from tactics:")
    print (r)

    sol = Solver()
    sol.add(r[0])
    print("start solver")
    res = sol.check()
    print(res)
    
    # print("Model for the original goal:")
    # print (r.convert_model(sol.model())) #buggato anche se in documentazione viene detto di usarlo così, va in errore
    if res == sat and needWitness == True:
      print("Model for the subgoal (witness):")
      witness = sol.model()
      print (witness)
      #isValidWithNeg(witness, qPos, qNeg, x_vars)
  else:
    print("start solver")
    res = s.check()
    print(res)
    if res == sat and needWitness == True:
      print("Model (witness):")
      witness = s.model()
      print(witness)
      #isValidWithNeg(witness, qPos, qNeg, x_vars)


  
def checkMLP(q, h_size, needWitness=True, magnitude = 0, withTactics=False):
  if withTactics==True:
    print("using tactics")
    s = Goal()
  else:
    s = Solver()
  
  in_size = 28*28

  x_vars = [Int(f'x_{i}') for i in range(in_size)] #create variables (one per each input pixel: 28*28=784)
  #s.add(Sum([ x_vars[i] for i in range(in_size)]) >= 20000)
  for x in x_vars:
    s.add(x >= 0)
    s.add(x <= 1)
  if (magnitude>0): s.add(Sum([x_vars[i] for i in range(len(x_vars))]) > magnitude) #magnitude per forzare non-trivial solutions

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
      weighted_sum = Sum([ x_vars[i] * RealVal(W_1[i][j].item()) for i in range(in_size)])
      y1_j = weighted_sum + RealVal(b_1[j].item())
      y1.append(y1_j)

    # ReLU - second layer
    print("encode second layer")
    y2 = [If(y1[j]>0, y1[j], 0) for j in range(h_size)] #relu encoding

    # linear combination - third layer
    print("encode third layer")
    y3 = Sum([RealVal(W_2[j].item()) * y2[j] for j in range(h_size)]) + RealVal(b_2)

    # add inequality to the system
    s.add(y3 >= 0)

  print("SMT encoding completed")
  # try:
  #   project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
  #   saveSolver(s, os.path.join(project_root, 'logs', 'solver_MLP.smt2'))
  # except Exception as e:
  #   print(f"Could not write solver to file: {e}")
  # return
  
  if withTactics==True:
    t1 = Tactic('simplify')
    t2 = Tactic('solve-eqs')
    t  = Then(t1, t2)
    r = t(s)
    print("result from tactics:")
    print (r)

    sol = Solver()
    sol.add(r[0])
    print("start solver")
    res = sol.check()
    print(res)
    
    # print("Model for the original goal:")
    # print (r.convert_model(sol.model())) #buggato anche se in documentazione viene detto di usarlo così, va in errore
    if res == sat and needWitness == True:
      print("Model for the subgoal (witness):")
      witness = sol.model()
      print (witness)
      #isValidWithNeg(witness, qPos, qNeg, x_vars)
  else:
    print("start solver")
    res = s.check()
    print(res)
    if res == sat and needWitness == True:
      print("Model (witness):")
      witness = s.model()
      print(witness)
      #isValidWithNeg(witness, qPos, qNeg, x_vars)


def generateSimpleProgram(q, mode, in_size=10, h_size=6, dom="integer"):
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
      saveSolver(s, os.path.join(project_root, 'logs', 'simple.smt2'))
    except Exception as e:
      print(f"Could not write solver to file: {e}")
    return
  
  elif (mode == 'solve'):
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