from src.utils import getModels
from src.marabou.marabou_encoder import *
from src.marabou.marabou_solver import *
from query_to_network_encoder import *
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENCODINGS_DIR = ROOT / "generated_encodings" / "ONNX"
ENCODINGS_DIR.mkdir(parents=True, exist_ok=True)




def main():
    models = getModels(modelType='MLP')

    query = [
        [[models[0], models[1]], [1, 1]],
        [[models[0], models[2]], [1, 0]]
    ]
    tbox = [
        [[models[0], models[1]], [0, 0]],
        [[models[0], models[2]], [0, 1]]
    ]
    setEncodingParameters(tbox, query)

    fileName = "mergedClassifiersEncoding"
    onnx_path = ENCODINGS_DIR / f"{fileName}.onnx"

    #qui sotto implementazione se il parser ONNX di Marabou supportasse Sign
    # model = networkEncoder()
    # writeONNX(model, onnx_path)
    # solveONNX(onnx_path)

    # qui sotto implementazione per supporto Sign
    mergedClassifiers = mergeClassifiers()
    writeONNX(mergedClassifiers, onnx_path)

    encodingWeights = getOntologyQueryEncodingWeights()

    network, outputVars = addOntologyAndQueryEncoding(onnx_path, encodingWeights)
    solveNetwork(network, outputVars)


if __name__ == "__main__":
    main()
