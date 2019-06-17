import numpy as np

def sigmoid(x):
    return np.power(1+np.exp(-x), -1)
