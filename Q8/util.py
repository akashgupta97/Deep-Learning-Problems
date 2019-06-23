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

def softmax(x):
    xexp = np.exp(x)
    esum = np.sum(xexp)
    return xexp/esum

def rmsprop(self, dw, db, neta, b1=.9, b2=.0, e=1e-8):
    for wpi, g in dw.items():
        self.m[wpi] = b1 * self.m[wpi] + (1 - b1) * np.square(g)
        self.w[wpi] -= neta * np.divide(g, (np.sqrt(self.m[wpi]) + e))
    for wpi in db:
        self.b[wpi] -= neta * db[wpi]
    return
