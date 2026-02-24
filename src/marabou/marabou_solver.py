from maraboupy import Marabou
from maraboupy import MarabouCore
from maraboupy.Marabou import createOptions

def solveNNET(nnetFile):
    nnet0 = "generated_encodings/NNET/codifica_0.nnet"
    nnet1 = "generated_encodings/NNET/codifica_1.nnet"

    network0 = Marabou.read_nnet(nnet0)
    network1 = Marabou.read_nnet(nnet1)

    network = Marabou.read_nnet(nnetFile)

    ret = network.solve()

    print("Network result:", ret[0])
    print("Network solution:", ret[1])
    print("Network stats", ret[2])
    #print("Network 0 numsplits", ret[2].getNumSplits())
    #print("Network 0 totaltime", ret[2].getTotalTime())
    #print("Network 1 vals:", vals1)
    #print("Network 1 stats:", stats1)

    query0 = network0.getInputQuery()
    query1 = network1.getInputQuery()
    #impossibile concatenare queste due reti usando funzioni native di marabou

def solveONNX(onnxFile):
    network = Marabou.read_onnx(onnxFile)

    inputVars = network.inputVars[0].flatten()
    outputVars = network.outputVars[0].flatten()

    for i in range(len(inputVars)):
        network.setLowerBound(inputVars[i], 0.0)
        network.setUpperBound(inputVars[i], 1.0)

    network.setLowerBound(outputVars[0], 0)

    ret = network.solve()
    print("Network result:", ret[0])
    print("Network solution:", ret[1])
    print("Network stats", ret[2])
