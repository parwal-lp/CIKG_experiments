#codifico la query Zero and One in una unica neural network
import numpy as np
from src.z3.z3_solver import *
from src.classifiers.train import *


'''
Funzione che crea una rete che dice sì se e solo se tutti i modelli in input dicono sì, altrimenti dice no
In particolare, per supportare gli atomi negati, ogni modello è specificato in coppia con una flag (True se il modello va considerato positivo, False se va negato)
'''
def andEncoder(predicates):
    input_size = 28*28
    single_model_h_size = 16
    out_size = len(predicates)
    combined_h_size = single_model_h_size*out_size
    zero_array = np.zeros([1, single_model_h_size])

    #dichiaro i parametri che verranno usati per la rete combinata finale
    Wcomb_1 = np.empty((combined_h_size, input_size))
    bcomb_1 = np.empty(combined_h_size)
    Wcomb_2 = np.empty((out_size, combined_h_size))
    bcomb_2 = np.empty(out_size)

    for index, (model, sign) in enumerate(predicates):
        W_1 = list(model.parameters())[0]
        b_1 = list(model.parameters())[1]
        W_2 = list(model.parameters())[2]
        b_2 = list(model.parameters())[3]

        row_start = index * single_model_h_size
        row_end = row_start + single_model_h_size

        Wcomb_1[row_start:row_end, :] = W_1.detach().cpu().numpy()
        bcomb_1[row_start:row_end] = b_1.detach().cpu().numpy()

        Wcomb_2[index, row_start:row_end] = W_2.detach().cpu().numpy().reshape(-1)
        bcomb_2[index] = b_2.detach().cpu().numpy().reshape(-1)[0]

    combinedModel = FlexMLP(input_size, combined_h_size, out_size)
    with torch.no_grad():
        combinedModel.fc1.weight.copy_(torch.tensor(Wcomb_1))
        combinedModel.fc1.bias.copy_(torch.tensor(bcomb_1))
        combinedModel.fc2.weight.copy_(torch.tensor(Wcomb_2))
        combinedModel.fc2.bias.copy_(torch.tensor(bcomb_2))
        combinedModel.fc2.weight.mul_(-1)
        combinedModel.fc2.bias.mul_(-1)
        combinedModel.fc3.weight.copy_(-torch.ones(out_size))

    return combinedModel