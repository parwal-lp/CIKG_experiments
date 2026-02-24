import argparse
from src.z3.z3_solver import *
from src.z3.z3_encoder import *
from src.classifiers.train import *
from src.utils import getModels

def main(args):
    #carico modelli
    models, _ = getModels(modelType='SLP', testFlag=False)

    #definisco ontologia
    inclAxioms = []
    disjAxioms = [(models[2], models[3])]

    #definisco query
    posConcepts = [models[1], models[2]]
    negConcepts = []
    
    #chiamo solver
    configureSolver(multithread=False)
    # start_time = time.time()
    # checkMLP(posConcepts, h_size=16, needWitness=True, magnitude=0, withTactics=False)
    # checkSLPwithNeg(posConcepts, negConcepts, inclAxioms, disjAxioms, needWitness=True, magnitude=0, withTactics=True)
    # generateSimpleProgram(posConcepts, 
    #                       mode=args.mode, 
    #                       in_size=args.in_size, 
    #                       h_size=args.h_size, 
    #                       dom=args.dom,
    #                       tactics=args.tactics)
    generateProgram(posConcepts, modelType='SLP')
    # end_time = time.time()

    #riporto tempistiche del solver
    # elapsed = end_time - start_time
    # print(f"Execution time: {elapsed:.4f}s")

def parse_args():
    parser = argparse.ArgumentParser(description="HKB experiments CLI")
    parser.add_argument("--mode", choices=["print", "solve"], default="print", help="Modalità di esecuzione (genera programma .smt2, o lo risolve con API python)")
    parser.add_argument("--in-size", type=int, default=50, dest="in_size", help="Dimensione input layer")
    parser.add_argument("--h-size", type=int, default=16, dest="h_size", help="Dimensione hidden layer")
    parser.add_argument("--dom", choices=["integer", "real"], default="integer", help="Dominio delle variabili")
    parser.add_argument("--tactics", choices=["yes", "no"], default="no", help="Semplificazione pre-solving")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)

