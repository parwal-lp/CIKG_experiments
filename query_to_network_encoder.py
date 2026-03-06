import numpy as np
from src.z3.z3_solver import *
from src.classifiers.train import *

'''
FUNZIONE DEPRECATA
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

        for row in range(W1_row_start, W1_row_end):
            for col in range(W1_columns):
                Wcomb_1[row, col] = W_1.detach().cpu().numpy()[row - W1_row_start, col]

        for pos in range(b1_idx_start, b1_idx_end):
            bcomb_1[pos] = b_1.detach().cpu().numpy()[pos - b1_idx_start]

        for col in range(W2_col_start, W2_col_end):
            Wcomb_2[W2_row, col] = W_2.detach().cpu().numpy()[0, col - W2_col_start]

        bcomb_2[index] = b_2.detach().cpu().numpy()[0]

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

'''
setto i parametri globali che verranno usati per costruire la rete finale
- axioms: lista degli assiomi della tbox
    questo parametro ha la forma 
    [ 
        [(atomi dell'assioma), (segni dell'assioma)], 
        ... 
    ]
    EXAMPLE: [[(a,b), (0,1)], [(a,c), (0,0)]] 
    means (!a OR b) equiv to (a->b) inclusion axiom, (!a OR !c) equiv to (a->!c) disjointness axiom
    numero di assiomi = O = len(axioms)
- query: la query da verificare
    questo parametro ha la forma 
    [ 
        [ (lista di atomi del disgiunto 0), (lista di segni del disgiunto 0) ],
        [ (atomi disgiunto 1), (segni disgiunto 1) ],
        ..., 
        [ (atomi disj D), (segni disj D) ]
    ]
    EXAMPLE: [ [(a,b,c), (1,0,1)], [(d),(0)] ] means (a AND !b AND c) OR (!d)
    numero di disgiunti = D = len(query)

TODO: idealmente dovrei poter passare la tbox e la query sparql
poi questa funzione dovrebbe chiamare tutti i vari metodi che formattano la tbox nel formato axioms e la querysparql nel formato query
'''
def setEncodingParameters(tbox, querySparql):
    global classifiers, axioms, query

    axioms = tbox
    #idealmente axioms = formatAxioms(tbox)

    query = querySparql
    #idealmente query = formatQuery(querySparql)

    classifiers = getAllClassifiers()

'''genera una rete che fa il merge di tutti i classificatori raccolti in classifiers
dando in output un vettore che contiene l'output di ciascun classificatore nell'ordine di input'''
def mergeClassifiers():
    input_size = 28*28
    single_model_h_size = 16
    num_classifiers = len(classifiers)
    combined_h_size = single_model_h_size*num_classifiers

    #dichiaro i parametri che verranno usati per la rete combinata finale
    Wcomb_1 = np.zeros((combined_h_size, input_size))
    bcomb_1 = np.zeros(combined_h_size)
    Wcomb_2 = np.zeros((num_classifiers, combined_h_size))
    bcomb_2 = np.zeros(num_classifiers)

    for index, model in enumerate(classifiers):
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

        for row in range(W1_row_start, W1_row_end):
            for col in range(W1_columns):
                Wcomb_1[row, col] = W_1.detach().cpu().numpy()[row - W1_row_start, col]

        for pos in range(b1_idx_start, b1_idx_end):
            bcomb_1[pos] = b_1.detach().cpu().numpy()[pos - b1_idx_start]

        for col in range(W2_col_start, W2_col_end):
            Wcomb_2[W2_row, col] = W_2.detach().cpu().numpy()[0, col - W2_col_start]

        bcomb_2[index] = b_2.detach().cpu().numpy()[0]

    combinedModel = SimpleMLP(input_size, combined_h_size, num_classifiers)
    with torch.no_grad():
        combinedModel.fc1.weight.copy_(torch.tensor(Wcomb_1))
        combinedModel.fc1.bias.copy_(torch.tensor(bcomb_1))
        combinedModel.fc2.weight.copy_(torch.tensor(Wcomb_2))
        combinedModel.fc2.bias.copy_(torch.tensor(bcomb_2))

    return combinedModel

'''
data la query e la tbox fissate usando fissiamo i parametri globali
questa funzione costruisce una rete che codifica la query e la tbox
in modo che dato in input un vettore di output dei classificatori, dia in output un vettore di 2 elementi:
il primo elemento è >= 0 se e solo se l'input soddisfa tutti gli assiomi della tbox
il secondo elemento è >= 0 se e solo se l'input soddisfa la query
'''
def queryontologyEncoder():
    n_classifiers = len(classifiers)
    n_axioms = len(axioms)
    n_disj = len(query)
    '''fc1'''
    W1 = np.eye(n_classifiers)
    #bias zero in the first layer = no bias, settato a False nella definizione del modello

    '''fc2'''
    W2 = np.eye(n_classifiers)
    b2 = np.ones(n_classifiers)

    # '''fc2 - pass-through in {-1,+1} domain (la conversione a {0,1} non funziona con sign(0)=1 di Marabou)'''
    # W2 = np.eye(n_classifiers)
    # b2 = np.zeros(n_classifiers)

    '''fc3'''
    W3 = np.zeros((n_disj+n_axioms, n_classifiers))
    for row in range(0, n_axioms):
        for col in range(0, n_classifiers):
            if (classifiers[col] not in axioms[row][0]): #il classificatore associato al nodo corrente (col) non è presente nel vincolo corrente (row)
                W3[row, col] = 0
            else:
                if (signOf(classifiers[col], axioms[row]) == 1): #il classificatore associato al nodo corrente (col) è presente nel vincolo corrente (row) positivamente
                    W3[row, col] = 1
                elif (signOf(classifiers[col], axioms[row]) == 0): #il classificatore associato al nodo corrente (col) è presente nel vincolo corrente (row) negato
                    W3[row, col] = -1
    for row in range(n_axioms, n_disj+n_axioms):
        rebased_row = row - n_axioms
        for col in range(0, n_classifiers):
            if (classifiers[col] not in query[rebased_row][0]): #il classificatore associato al nodo corrente (col) non è presente nel disgiunto corrente (row)
                W3[row, col] = 0
            else:
                if (signOf(classifiers[col], query[rebased_row]) == 1): #il classificatore associato al nodo corrente (col) è presente nel disgiunto corrente (row) positivamente
                    W3[row, col] = 1
                elif (signOf(classifiers[col], query[rebased_row]) == 0): #il classificatore associato al nodo corrente (col) è presente nel disgiunto corrente (row) negato
                    W3[row, col] = -1

    b3 = np.zeros(n_disj+n_axioms)
    for i in range(0, n_axioms):
        b3[i] = axioms[i][1].count(0) #numero di atomi negati nel vincolo corrente
    for i in range(n_axioms, n_disj+n_axioms):
        rebased_i = i - n_axioms
        b3[i] = query[rebased_i][1].count(0) - len(query[rebased_i][1]) #numero atomi negati nel disgiunto meno atomi totali del disgiunto -> uguale a - numero di atomi positivi? (-query[i].count(1)??)

    '''fc4'''
    W4 = np.zeros((2, n_disj+n_axioms))
    for col in range(0, n_axioms):
        W4[0][col] = 1
    for col in range(n_axioms, n_disj+n_axioms):
        W4[1][col] = 1
    b4 = np.zeros(2)
    b4[0] = -n_axioms
    b4[1] = len(query)-1

    encodedModel = OntologyAndQueryNetwork(n_classifiers, n_disj, n_axioms, out_size=2)
    with torch.no_grad():
        encodedModel.fc1.weight.copy_(torch.tensor(W1))
        #encodedModel.fc1.bias.copy_(torch.tensor(b1)) #bias zero in the first layer = no bias, settato a False nella definizione del modello
        encodedModel.fc2.weight.copy_(torch.tensor(W2))
        encodedModel.fc2.bias.copy_(torch.tensor(b2))
        encodedModel.fc3.weight.copy_(torch.tensor(W3))
        encodedModel.fc3.bias.copy_(torch.tensor(b3))
        encodedModel.fc4.weight.copy_(torch.tensor(W4))
        encodedModel.fc4.bias.copy_(torch.tensor(b4))
    return encodedModel

'''
dato un concetto (classificatore) e un'espressione (assioma o disgiunto)
restituisce il segno associato a quel concetto in quell'espressione, se è presente
'''
def signOf(concept, expression):
    atoms = expression[0]
    signs = expression[1]
    if concept not in atoms:
        return "error: concept not in expression"
    else:
        index = atoms.index(concept)
        sign = signs[index]
    return sign


'''
date le strutture dati axioms e query
ricava tutti i classificatori che compaiono nell'una o nell'altra, o in entrambe, senza ripetizioni
'''
def getAllClassifiers():
    classifiers = set()
    for pair in axioms:
        axiom = pair[0]
        for atom in axiom:
            classifiers.add(atom)
    for pair in query:
        disj = pair[0]
        for atom in disj:
            classifiers.add(atom)
    return list(classifiers)

'''
fa la concatenzaione della rete che fa il merge di tutti i classificatori
con le rete che codifica la query e la tbox, in modo da ottenere l'encoding finale di tutto il problema
'''
def networkEncoder():
    mergedClassifiers = mergeClassifiers()
    encodedQueryAndOntology = queryontologyEncoder()
    finalEncoding = finalEncodedNetwork(mergedClassifiers, encodedQueryAndOntology)
    return finalEncoding

'''
Restituisce solo la rete dei classificatori mergiati (SimpleMLP con ReLU)
e i pesi della rete di encoding ontologia+query, separatamente.
Questo serve per poter esportare solo la parte ReLU come ONNX
e costruire manualmente la parte con Sign in Marabou.
'''
def getEncoderComponents():
    merged = mergeClassifiers()
    encodedModel = queryontologyEncoder()
    # Estrai i pesi dell'OntologyAndQueryNetwork
    weights = {
        'W1': list(encodedModel.parameters())[0].detach().cpu().numpy(),  # fc1.weight (no bias)
        'W2': list(encodedModel.parameters())[1].detach().cpu().numpy(),  # fc2.weight
        'b2': list(encodedModel.parameters())[2].detach().cpu().numpy(),  # fc2.bias
        'W3': list(encodedModel.parameters())[3].detach().cpu().numpy(),  # fc3.weight
        'b3': list(encodedModel.parameters())[4].detach().cpu().numpy(),  # fc3.bias
        'W4': list(encodedModel.parameters())[5].detach().cpu().numpy(),  # fc4.weight
        'b4': list(encodedModel.parameters())[6].detach().cpu().numpy(),  # fc4.bias
    }
    return merged, weights


'''
data la query e la tbox fissate usando fissiamo i parametri globali
questa funzione costruisce le matrici W e i bias b dei layer che servono a fare l'encoding di query e ontologi
'''
def getOntologyQueryEncodingWeights():
    n_classifiers = len(classifiers)
    n_axioms = len(axioms)
    n_disj = len(query)

    '''fc1'''
    W1 = np.zeros((n_disj+n_axioms, n_classifiers))
    for row in range(0, n_axioms):
        for col in range(0, n_classifiers):
            if (classifiers[col] not in axioms[row][0]): #il classificatore associato al nodo corrente (col) non è presente nel vincolo corrente (row)
                W1[row, col] = 0
            else:
                if (signOf(classifiers[col], axioms[row]) == 1): #il classificatore associato al nodo corrente (col) è presente nel vincolo corrente (row) positivamente
                    W1[row, col] = 1
                elif (signOf(classifiers[col], axioms[row]) == 0): #il classificatore associato al nodo corrente (col) è presente nel vincolo corrente (row) negato
                    W1[row, col] = -1
    for row in range(n_axioms, n_disj+n_axioms):
        rebased_row = row - n_axioms
        for col in range(0, n_classifiers):
            if (classifiers[col] not in query[rebased_row][0]): #il classificatore associato al nodo corrente (col) non è presente nel disgiunto corrente (row)
                W1[row, col] = 0
            else:
                if (signOf(classifiers[col], query[rebased_row]) == 1): #il classificatore associato al nodo corrente (col) è presente nel disgiunto corrente (row) positivamente
                    W1[row, col] = 1
                elif (signOf(classifiers[col], query[rebased_row]) == 0): #il classificatore associato al nodo corrente (col) è presente nel disgiunto corrente (row) negato
                    W1[row, col] = -1

    b1 = np.zeros(n_disj+n_axioms)
    for i in range(0, n_axioms):
        b1[i] = 0 #no bias
    for i in range(n_axioms, n_disj+n_axioms):
        rebased_i = i - n_axioms
        b1[i] = len(query[rebased_i][1]) #numero atomi nel disgiunto

    '''fc2'''
    W2 = np.zeros((2, n_disj+n_axioms))
    for col in range(0, n_axioms):
        W2[0][col] = 1
    for col in range(n_axioms, n_disj+n_axioms):
        W2[1][col] = 1
    b2 = np.zeros(2)
    b2[0] = -n_axioms
    b2[1] = len(query)-1

    weightsDict = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

    return weightsDict