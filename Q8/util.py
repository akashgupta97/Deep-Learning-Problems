import numpy as np

def sigmoid(x):
    return np.power(1+np.exp(-x), -1)

def dsigmoid(x):
    t=sigmoid(x)
    return (1-t)*t

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return  1-np.square(np.tanh(x))
