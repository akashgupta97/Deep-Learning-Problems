import numpy as np
import pickle
from util import sigmoid, dsigmoid, softmax, tanh, dtanh, rmsprop, adam

class gru:

    def __init__(self, i_size, h_size, o_size, optimize='rmsprop', wb=None):

        self.optimize = optimize
        self.names = {0:'ur',1:'wr', 2:'uz', 3:'wz', 4:'u_h', 5:'w_h', 6:'wo'}
