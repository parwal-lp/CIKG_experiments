import sys
import numpy as np

## %
# Path to Marabou folder if you did not export it

# sys.path.append('/home/USER/git/Marabou')
from maraboupy import Marabou

nnet0 = "codifica_0.nnet"
nnet1 = "codifica_1.nnet"

encoding_0 = Marabou.read_nnet(nnet0)
encoding_1 = Marabou.read_nnet(nnet1)

vals1, stats1 = encoding_1.solve()