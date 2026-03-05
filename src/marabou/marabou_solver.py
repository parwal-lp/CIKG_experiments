from maraboupy import Marabou
from maraboupy import MarabouCore
from maraboupy import MarabouUtils
from maraboupy.Marabou import createOptions
import numpy as np

def solveNNET(nnetFilePath):
    nnet0 = "generated_encodings/NNET/codifica_0.nnet"
    nnet1 = "generated_encodings/NNET/codifica_1.nnet"

    network0 = Marabou.read_nnet(nnet0)
    network1 = Marabou.read_nnet(nnet1)

    network = Marabou.read_nnet(nnetFilePath)

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

def solveONNX(onnxFilePath):
    network = Marabou.read_onnx(onnxFilePath)

    inputVars = network.inputVars[0].flatten()
    outputVars = network.outputVars[0].flatten()

    print("input vars", len(inputVars))
    print("output vars", len(outputVars))

    for i in range(len(inputVars)):
        network.setLowerBound(inputVars[i], 0.0)
        network.setUpperBound(inputVars[i], 1.0)

    network.setLowerBound(outputVars[0], 0) #devono essere soddisfatti tutti gli assiomi della tbox
    network.setLowerBound(outputVars[1], 0) #deve essere soddisfatta la query

    ret = network.solve()
    # print("Network result:", ret[0])
    # print("Network solution:", ret[1])
    # print("Network stats", ret[2])

# da qui sotto iniziano implementazioni dovute al mancato supporto di Sign nel parser ONNX di Marabou

def _addLinearLayer(network, inputVars, W, b=None):
    """Aggiunge un layer lineare y = W * x + b alla rete Marabou.
    Restituisce le variabili di output."""
    out_size, in_size = W.shape
    outputVars = []
    for i in range(out_size):
        outVar = network.getNewVariable()
        outputVars.append(outVar)
        eq = MarabouUtils.Equation(MarabouCore.Equation.EquationType.EQ)
        for j in range(in_size):
            eq.addAddend(W[i, j], inputVars[j])
        eq.addAddend(-1.0, outVar)
        eq.setScalar(-(b[i] if b is not None else 0.0))
        network.addEquation(eq)
    return outputVars


def _addSignLayer(network, inputVars):
    """Aggiunge un layer Sign alla rete Marabou usando addSignConstraint.
    Restituisce le variabili di output."""
    outputVars = []
    for inVar in inputVars:
        outVar = network.getNewVariable()
        outputVars.append(outVar)
        network.addSignConstraint(inVar, outVar)
    return outputVars


def solveONNXWithSignSupport(onnxFilePath, ontologyWeights):
    """Carica solo il mergedClassifiers da ONNX, poi aggiunge manualmente
    i layer dell'OntologyAndQueryNetwork con Sign via API Marabou.

    ontologyWeights è un dict con chiavi: W1, W2, b2, W3, b3, W4, b4
    corrispondenti ai pesi dell'OntologyAndQueryNetwork.
    """
    network = Marabou.read_onnx(onnxFilePath)

    inputVars = network.inputVars[0].flatten()
    # Gli output dell'ONNX sono gli output del mergedClassifiers
    classifierOutputVars = network.outputVars[0].flatten()

    n_inputs = len(inputVars)
    n_classifiers = len(classifierOutputVars)
    print(f"input vars: {n_inputs}")
    print(f"classifier output vars: {n_classifiers}")

    # Vincoli sugli input: pixel nell'intervallo [0, 1]
    for i in range(n_inputs):
        network.setLowerBound(inputVars[i], 0.0)
        network.setUpperBound(inputVars[i], 1.0)

    W1 = ontologyWeights['W1']
    W2 = ontologyWeights['W2']
    b2 = ontologyWeights['b2']
    W3 = ontologyWeights['W3']
    b3 = ontologyWeights['b3']
    W4 = ontologyWeights['W4']
    b4 = ontologyWeights['b4']

    # Layer 1: linear (no bias) + sign
    layer1_linear = _addLinearLayer(network, classifierOutputVars, W1)
    layer1_sign = _addSignLayer(network, layer1_linear)

    # Layer 2: linear (with bias) + sign
    layer2_linear = _addLinearLayer(network, layer1_sign, W2, b2)
    layer2_sign = _addSignLayer(network, layer2_linear)

    # Layer 3: linear (with bias) + sign
    layer3_linear = _addLinearLayer(network, layer2_sign, W3, b3)
    layer3_sign = _addSignLayer(network, layer3_linear)

    # Layer 4: linear (with bias) — no activation, these are the final outputs
    finalOutputVars = _addLinearLayer(network, layer3_sign, W4, b4)

    print(f"final output vars: {len(finalOutputVars)}")

    # Vincoli sull'output:
    # outputVars[0] >= 0: tutti gli assiomi della tbox soddisfatti
    # outputVars[1] >= 0: query soddisfatta
    network.setLowerBound(finalOutputVars[0], 0)
    network.setLowerBound(finalOutputVars[1], 0)

    ret = network.solve()
    print("Result:", ret[0])
    if ret[1]:
        print("Solution found")
    else:
        print("No solution")
