from src.solver import *
from src.train import *
import time
import random
from itertools import combinations
from collections import defaultdict

def getModels(modelType, testFlag=False):
    device = detectDevice()
    # [train_dataset, train_loader] = setTrainDatasets()

    SLPmodels = loadModels(modelType, device)
    if(testFlag): testModels(SLPmodels, test_loader, device)

    SLPeven = SimpleSLP().to(device)
    SLPeven.load_state_dict(torch.load(f'./models/SLP/model_[0, 2, 4, 6, 8]', weights_only=True, map_location=device))
    if(testFlag): 
        [test_dataset, test_loader, posTests] = setTestDatasets()
        test(test_loader, SLPeven, [2,4,6,8], device)
    return SLPmodels, SLPeven

def stressTest(SLPmodels):
    max_query_len = 6
    max_axioms = 6
    samples_per_combination = 1
    results = defaultdict(list)

    #DEFINE 6 ONTOLOGIES, one per each length
    axioms_combination = list(combinations(SLPmodels, 2))
    ontologies = []
    ontologies.append([]) # the ontology at index 0 is an empty ontology, i.e., with 0 axioms
    for axiom_len in range(max_axioms):
        sample_axioms = random.sample(axioms_combination, axiom_len)
        ontologies.append(sample_axioms)

    #START THE TEST
    for query_len in range(1, max_query_len + 1): #per each length of query
        query_combinations = list(combinations(SLPmodels, query_len))
        sample_queries = random.sample(query_combinations, min(samples_per_combination, len(query_combinations))) #sample 10 queries of the current length

        for axiom_set, num_axioms in zip(ontologies, range(0, len(ontologies))): #per each length of axiom
            for q in sample_queries: #execute all the 10 queries of length query_len with ontology of num_axioms axioms
                q = list(q)

                start_time = time.time()
                checkSLP(q, incT=sample_axioms, disjT=[], needWitness=False) #call to the function that encodes and answers the query
                end_time = time.time()

                elapsed = end_time - start_time
                print(f"QueryLen: {query_len}, NumAxioms: {num_axioms}, Time: {elapsed:.4f}s")
                results[(query_len, num_axioms)].append(elapsed)

    # Report average times
    print("\n===== AVERAGE TIMES =====")
    for (qlen, naxioms), times in sorted(results.items()):
        avg = sum(times) / len(times)
        print(f"QueryLen {qlen}, NumAxioms {naxioms}: Avg time = {avg:.4f}s")


### Asking a specific query on a specific ontology
# Edit the following code to define a custom query and a custom ontology, then run the snippet to compute the answer.
# - define posConcepts as the list of concepts appearing non-negated in the query
# - define negConcepts as the list of concepts appearing negated in the query
# - add in inclAxioms a pair (a, b) to define an inclusion assertion from a to b
# - add in disjAxioms a pair (a, b) to define a disjointness assertion between a and b
def main():
    #carico modelli
    SLPmodels, SLPeven = getModels(modelType='SLP', testFlag=False)

    #definisco ontologia
    inclAxioms = [(SLPmodels[2], SLPeven)]
    disjAxioms = []

    #definisco query
    posConcepts = [SLPmodels[1], SLPmodels[3]]
    negConcepts = []
    
    #chiamo solver
    start_time = time.time()
    checkSLPwithNeg(posConcepts, negConcepts, inclAxioms, disjAxioms, needWitness=True, magnitude=50)
    end_time = time.time()

    #riporto tempistiche del solver
    elapsed = end_time - start_time
    print(f"Execution time: {elapsed:.4f}s")

if __name__ == "__main__":
    main()

