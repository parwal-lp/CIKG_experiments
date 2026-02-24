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
    query_size = len(predicates)
    combined_h_size = single_model_h_size*query_size

    #dichiaro i parametri che verranno usati per la rete combinata finale
    Wcomb_1 = np.zeros((combined_h_size, input_size))
    bcomb_1 = np.zeros(combined_h_size)
    Wcomb_2 = np.zeros((query_size, combined_h_size))
    bcomb_2 = np.zeros(query_size)

    for index, (model, sign) in enumerate(predicates):
        W_1 = list(model.parameters())[0]
        b_1 = list(model.parameters())[1]
        W_2 = list(model.parameters())[2]
        b_2 = list(model.parameters())[3]

        W1_row_start = index * single_model_h_size
        W1_row_end = W1_row_start + single_model_h_size
        W1_columns = input_size

        b1_idx_start = index * single_model_h_size
        b1_idx_end = b1_idx_start + single_model_h_size

        W2_row = index
        W2_col_start = index * single_model_h_size
        W2_col_end = W2_col_start + single_model_h_size

        b2_idx_start = index * query_size
        b2_idx_end = b2_idx_start + query_size

        for row in range(W1_row_start, W1_row_end):
            for col in range(W1_columns):
                Wcomb_1[row, col] = W_1.detach().cpu().numpy()[row - W1_row_start, col]

        for pos in range(b1_idx_start, b1_idx_end):
            bcomb_1[pos] = b_1.detach().cpu().numpy()[pos - b1_idx_start]

        for col in range(W2_col_start, W2_col_end):
            Wcomb_2[W2_row, col] = W_2.detach().cpu().numpy()[0, col - W2_col_start]

        for pos in range(b2_idx_start, b2_idx_end):
            bcomb_2[pos] = b_2.detach().cpu().numpy()[pos - b2_idx_start]

    combinedModel = FlexMLP(input_size, combined_h_size, query_size)
    with torch.no_grad():
        combinedModel.fc1.weight.copy_(torch.tensor(Wcomb_1))
        combinedModel.fc1.bias.copy_(torch.tensor(bcomb_1))
        combinedModel.fc2.weight.copy_(torch.tensor(Wcomb_2))
        combinedModel.fc2.bias.copy_(torch.tensor(bcomb_2))
        combinedModel.fc2.weight.mul_(-1)
        combinedModel.fc2.bias.mul_(-1)
        combinedModel.fc3.weight.copy_(-torch.ones(query_size))

    return combinedModel