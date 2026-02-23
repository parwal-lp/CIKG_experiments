import sys
import numpy as np

from maraboupy import Marabou
from maraboupy import MarabouCore

nnet0 = "codifica_0.nnet"
nnet1 = "codifica_1.nnet"

network0 = Marabou.read_nnet(nnet0)
network1 = Marabou.read_nnet(nnet1)

vals0, stats0 = network0.solve()
vals1, stats1 = network1.solve()

print("Network 0 vals:", vals0)
print("Network 0 stats:", stats0)
print("Network 1 vals:", vals1)
print("Network 1 stats:", stats1)



# query = MarabouCore.InputQuery()
# query.setNumberOfVariables(28*28)
# #query0 = MarabouCore.InputQuery()
# MarabouCore.createInputQuery(query, nnet0, '')

# query.addEquation(query0)

# ret = query.solve()

# print(ret)
