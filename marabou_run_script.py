from src.utils import getModels
from src.marabou.marabou_encoder import encodeModelToNNet

def main(args):
    models = getModels(modelType='MLP')

    for model in models:
        encodeModelToNNet(model, f'codifica_{models.index(model)}.nnet')


if __name__ == "__main__":
    main()


