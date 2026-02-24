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

    for index, model in enumerate(models):
        # encodeModelToNNet(model, f'codifica_{models.index(model)}.nnet')
        print(f"_______________MODEL {index}_______________")
        print(f"matrice pesi W0 ha dimensioni {list(model.parameters())[0].shape}")
        print(f"vettore bias b0 ha dimensioni {list(model.parameters())[1].shape}")
        print(f"matrice pesi W1 ha dimensioni {list(model.parameters())[2].shape}")
        print(f"vettore bias b1 ha dimensioni {list(model.parameters())[3].shape}")

    # query = [(models[0], True), (models[1], True)]

    # model = andEncoder(query)

    # fileName = "0_and_1"
    # onnx_path = ENCODINGS_DIR / f"{fileName}.onnx"
    # writeONNX(model, onnx_path)

    # solveONNX(onnx_path)



if __name__ == "__main__":
    main()
