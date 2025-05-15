# CIKG_experiments
Proof of concept for the CIKG paper

This repository is structured as follows:
- the ./data folder contains the datasets used for training and testing the models
- under the folder ./src there are all the functions defined to train the models and to define the SMT encoding
- under the ./models folder there can be found all the pre-trained weigths for all the models used in the experiment
- the file main.ipynb at the root of this repository contains all the steps necessary to run the experiments

To run the experiments, download or clone the repository, and then follow the instructions in main.ipynb

## Requirements
Running the code in main.ipynb requires PyTorch and Z3Py to be installed.
This code was tested using PyTorch 2.7.0+cu128 and Z3 version 4.14.1 - 64 bit.