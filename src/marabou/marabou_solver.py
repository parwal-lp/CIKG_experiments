from maraboupy import Marabou
from maraboupy import MarabouCore
from maraboupy import MarabouUtils
from maraboupy.Marabou import createOptions
import numpy as np

'''
    dato NNET file chiama il solver di Marabou su quella rete
'''
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

'''
    dato ONNX file chiama il solver di Marabou su quella rete
'''
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

'''
-------------------------------------------------------------------------------
da qui sotto iniziano implementazioni dovute al mancato supporto di Sign
nel parser ONNX di Marabou e nel formato NNET
-------------------------------------------------------------------------------
'''

'''
    Aggiunge un layer lineare y = W * x + b alla rete in input
    Restituisce le variabili di output
'''
def addLinearCombination(network, inputVars, W, b=None):
    out_size = W.shape[0]
    outputVars = []

    for i in range(out_size):
        outVar = network.getNewVariable()
        outputVars.append(outVar)
        network.addEquality(
            vars   = list(inputVars) + [outVar],
            coeffs = list(W[i]) + [-1.0],
            scalar = -(b[i] if b is not None else 0.0)
        )
    print("ho aggiunto le variabili", outputVars)

    return outputVars

'''
    DEPRECATA - NON POSSO USARLA
    Aggiunge una activation function di tipo Sign (Marabou) alla rete in input.
    sign_marabou: sign(x<0)=-1, sign(x>=0)=+1 -> ATTENZIONE: mismatch con sign di pytorch sign(x<0)=-1, sign(x=0)=0, sign(x>0)=+1
    Restituisce le variabili di output
'''
def addSignActivation(network, inputVars):
    outputVars = []
    for inVar in inputVars:
        outVar = network.getNewVariable()
        outputVars.append(outVar)
        network.addSignConstraint(inVar, outVar)
    return outputVars

'''
    Aggiunge una activation function sign3 (three way sign) compatibile con PyTorch:
    sign3(x<0)=-1, sign3(x=0)=0, sign3(x>0)=+1
    Restituisce le variabili di output
    TODO not implemented yet
'''
def addSign3Activation(network, inputVars):

    # x > 0  →  y = 1
    # x = 0  →  y = 0
    # x < 0  →  y = -1

    outputVars = []

    for inVar in inputVars:
        outVar = network.getNewVariable()
        outputVars.append(outVar)

    return outputVars

'''
    chiama solve di Marabou su una rete costituita dal merge dei classificatori (preso da ONNX)
    seguito da layer costruiti manualmente dell'encoding query e tbox

    - encodingWeights è un dict con chiavi: W1, W2, b2, W3, b3, W4, b4 corrispondenti ai pesi dell'OntologyAndQueryNetwork
    - mergedClassifiersPath è il path del file ONNX che contiene solo la parte di rete che fa il merge dei classificatori
'''
def solveNetworkWithSign3(mergedClassifiersPath, encodingWeights):
    
    mergedClassifiersNetwork = Marabou.read_onnx(mergedClassifiersPath)

    inputVars = mergedClassifiersNetwork.inputVars[0].flatten()
    # Gli output dell'ONNX della rete merge dei classificatori
    mergedClassifiersOutputVars = mergedClassifiersNetwork.outputVars[0].flatten()

    n_inputs = len(inputVars)
    n_classifiers = len(mergedClassifiersOutputVars)
    print(f"input vars: {n_inputs}")
    print(f"number of classifiers in merged network: {n_classifiers}")

    # Vincoli sugli input: pixel nell'intervallo [0, 1]
    for i in range(n_inputs):
        mergedClassifiersNetwork.setLowerBound(inputVars[i], 0.0)
        mergedClassifiersNetwork.setUpperBound(inputVars[i], 1.0)

    W1 = encodingWeights['W1']
    W2 = encodingWeights['W2']
    b2 = encodingWeights['b2']
    W3 = encodingWeights['W3']
    b3 = encodingWeights['b3']
    W4 = encodingWeights['W4']
    b4 = encodingWeights['b4']

    finalNetwork = mergedClassifiersNetwork

    # Layer 1: linear (no bias) + sign
    layer1_linear = addLinearCombination(finalNetwork, mergedClassifiersOutputVars, W1)
    layer1_sign = addSign3Activation(finalNetwork, layer1_linear)

    # Layer 2: linear (with bias) + sign
    layer2_linear = addLinearCombination(finalNetwork, layer1_sign, W2, b2)
    layer2_sign = addSign3Activation(finalNetwork, layer2_linear)

    # Layer 3: linear (with bias) + sign
    layer3_linear = addLinearCombination(finalNetwork, layer2_sign, W3, b3)
    layer3_sign = addSign3Activation(finalNetwork, layer3_linear)

    # Layer 4: linear (with bias) and no activation, these are the final outputs
    finalOutputVars = addLinearCombination(finalNetwork, layer3_sign, W4, b4)

    print(f"final output vars: {len(finalOutputVars)}")

    # Vincoli sull'output:
    # outputVars[0] >= 0: tutti gli assiomi della tbox soddisfatti
    # outputVars[1] >= 0: query soddisfatta
    finalNetwork.setLowerBound(finalOutputVars[0], 0)
    finalNetwork.setLowerBound(finalOutputVars[1], 0)

    ret = finalNetwork.solve()
    print("Result:", ret[0])
    if ret[1]:
        print("Solution found")
    else:
        print("No solution")

'''
    data una rete che fa il merge di tutti i classificatori (da file ONNX), 
    accoda layer costruiti manualmente per fare l'encoding di query e tbox
    utilizzando la Sign con la semantica di Marabou

    - encodingWeights è un dict con chiavi: W1, b1, W2, b2 corrispondenti ai pesi dei layer che fanno l'encoding di query e ontologia
    - mergedClassifiersPath è il path del file ONNX che contiene solo la parte di rete che fa il merge dei classificatori
'''
def addOntologyAndQueryEncoding(mergedClassifiersPath, encodingWeights):
    
    mergedClassifiersNetwork = Marabou.read_onnx(mergedClassifiersPath)

    inputVars = mergedClassifiersNetwork.inputVars[0].flatten()
    # Gli output dell'ONNX della rete merge dei classificatori
    mergedClassifiersOutputVars = mergedClassifiersNetwork.outputVars[0].flatten()

    n_inputs = len(inputVars)
    n_classifiers = len(mergedClassifiersOutputVars)
    print(f"input vars: {n_inputs}")
    print(f"number of classifiers in merged network: {n_classifiers}")

    # Vincoli sugli input: pixel nell'intervallo [0, 1]
    for i in range(n_inputs):
        mergedClassifiersNetwork.setLowerBound(inputVars[i], 0.0)
        mergedClassifiersNetwork.setUpperBound(inputVars[i], 1.0)

    W1 = encodingWeights['W1']
    b1 = encodingWeights['b1']
    W2 = encodingWeights['W2']
    b2 = encodingWeights['b2']

    finalNetwork = mergedClassifiersNetwork

    # Layer 0: sign
    layer0_sign = addSignActivation(finalNetwork, mergedClassifiersOutputVars)

    # Layer 1: linear (with bias) + sign
    layer1_linear = addLinearCombination(finalNetwork, layer0_sign, W1, b1)
    layer1_sign = addSignActivation(finalNetwork, layer1_linear)

    # Layer 2: linear (with bias)
    finalOutputVars = addLinearCombination(finalNetwork, layer1_sign, W2, b2)

    print(f"final output vars: {len(finalOutputVars)}")

    # Vincoli sull'output:
    # outputVars[0] >= 0: tutti gli assiomi della tbox soddisfatti
    # outputVars[1] >= 0: query soddisfatta
    finalNetwork.setLowerBound(finalOutputVars[0], 0)
    finalNetwork.setLowerBound(finalOutputVars[1], 0)

    return finalNetwork, finalOutputVars



def solveNetwork(network, finalOutputVars):
    ret = network.solve()
    print("Result:", ret[0])
    if ret[1]:
        print("Solution found: ->", ret[1])
        print("output ontologia:",ret[1][finalOutputVars[0]])
        print("output query:", ret[1][finalOutputVars[1]])
    else:
        print("No solution")
