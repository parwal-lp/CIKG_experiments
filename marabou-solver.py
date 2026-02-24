import sys
import numpy as np

from maraboupy import Marabou
from maraboupy import MarabouCore
from maraboupy.Marabou import createOptions

# NNET version
'''
nnet0 = "codifica_0.nnet"
nnet1 = "codifica_1.nnet"

network0 = Marabou.read_nnet(nnet0)
network1 = Marabou.read_nnet(nnet1)

options = createOptions()

ret = network0.solve()
#vals1, stats1 = network1.solve()

print("Network 0 result:", ret[0])
print("Network 0 solution:", ret[1])
print("Network 0 stats", ret[2])
#print("Network 0 numsplits", ret[2].getNumSplits())
#print("Network 0 totaltime", ret[2].getTotalTime())
#print("Network 1 vals:", vals1)
#print("Network 1 stats:", stats1)

query0 = network0.getInputQuery()
query1 = network1.getInputQuery()
'''

# ONNX version
network = Marabou.read_onnx("codifica_0and1.onnx")

inputVars = network.inputVars[0].flatten()
outputVars = network.outputVars[0].flatten()

for i in range(len(inputVars)):
    network.setLowerBound(inputVars[i], 0.0)
    network.setUpperBound(inputVars[i], 1.0)

network.setLowerBound(outputVars[0], 0)

result = network.solve()
print(result[0])
print(result[1])
print(result[2])
