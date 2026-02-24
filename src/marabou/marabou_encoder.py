from src.z3.z3_solver import *
from src.classifiers.train import *

'''FUNZIONE PRESA DA REPO GITHUB NNET, NON MODIFICATA'''
def writeNNet(weights, biases, inputMins, inputMaxes, means, ranges, fileName):
    '''
    Write network data to the .nnet file format.

    Args:
        weights (list): Weight matrices in the network order 
        biases (list): Bias vectors in the network order
        inputMins (list): Minimum values for each input
        inputMaxes (list): Maximum values for each input
        means (list): Mean values for each input and a mean value for all outputs. Used to normalize inputs/outputs
        ranges (list): Range values for each input and a range value for all outputs. Used to normalize inputs/outputs
        fileName (str): File where the network will be written
    '''
    try:
        # Validate dimensions of weights and biases
        assert len(weights) == len(biases), "Number of weight matrices and bias vectors must match."
        
        # Open the file we wish to write
        with open(fileName, 'w') as f2:
            f2.write("// Neural Network File Format\n")

            numLayers = len(weights)
            inputSize = weights[0].shape[1]
            outputSize = len(biases[-1])
            maxLayerSize = max(inputSize, max(len(b) for b in biases))

            # Write network architecture info
            f2.write(f"{numLayers},{inputSize},{outputSize},{maxLayerSize},\n")
            f2.write(f"{inputSize}," + ",".join(str(len(b)) for b in biases) + ",\n")
            f2.write("0,\n")  # Unused flag

            # Write normalization information
            f2.write(",".join(map(str, inputMins)) + ",\n")
            f2.write(",".join(map(str, inputMaxes)) + ",\n")
            f2.write(",".join(map(str, means)) + ",\n")
            f2.write(",".join(map(str, ranges)) + ",\n")

            # Write weights and biases
            for w, b in zip(weights, biases):
                for i in range(w.shape[0]):
                    f2.write(",".join(f"{w[i, j]:.5e}" for j in range(w.shape[1])) + ",\n")
                for i in range(len(b)):
                    f2.write(f"{b[i]:.5e},\n")

    except Exception as e:
        print(f"Error writing NNet file: {e}")
        raise

def encodeModelToNNet(model, fileName):
    # Extract parameters and convert to CPU numpy arrays without altering shapes
    W_1 = list(model.parameters())[0].detach().cpu().numpy()  # shape: (out1, in)
    b_1 = list(model.parameters())[1].detach().cpu().numpy()  # shape: (out1,)
    W_2 = list(model.parameters())[2].detach().cpu().numpy()  # shape: (out2, out1)
    b_2 = list(model.parameters())[3].detach().cpu().numpy()  # shape: (out2,)

    # Prepare normalization arrays (includes one extra value for outputs)
    input_size = 28*28
    input_mins = [0] * input_size
    input_maxes = [1] * input_size
    means = [0] * input_size + [0]
    ranges = [1] * input_size + [1]

    writeNNet(
        weights=[W_1, W_2],
        biases=[b_1, b_2],
        inputMins=input_mins,
        inputMaxes=input_maxes,
        means=means,
        ranges=ranges,
        fileName=fileName
    )

def writeONNX(model, filePath):
    torch.onnx.export(model, torch.zeros(784), filePath)
    print(f"Model exported to ONNX format at {filePath}")