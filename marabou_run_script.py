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

    # query = [(models[0], True), (models[1], True), (models[2], False), (models[3], True), (models[4], False), (models[5], True), (models[6], False), (models[7], True), (models[8], False), (models[9], True)]

    # model = andEncoder(query)

    query = [
        [(models[1], models[2]), (1, 1)]
        [(models[0], models[1]), (1, 0)]
    ]
    tbox = [
        [(models[0], models[2]), (0, 1)]
        [(models[1], models[3]), (0, 0)]
    ]
    setEncodingParameters(query, tbox)
    model = networkEncoder()


    fileName = "megaEncodingTest"
    onnx_path = ENCODINGS_DIR / f"{fileName}.onnx"
    writeONNX(model, onnx_path)

    solveONNX(onnx_path)



if __name__ == "__main__":
    main()
