import numpy as np
import pickle
from util import sigmoid, dsigmoid, softmax, tanh, dtanh, rmsprop, adam

class gru:

    def __init__(self, i_size, h_size, o_size, optimize='rmsprop', wb=None):

        self.optimize = optimize
        self.names = {0:'ur',1:'wr', 2:'uz', 3:'wz', 4:'u_h', 5:'w_h', 6:'wo'}
        if wb:
            self.w, self.b = self.load_model(wb)
            self.h_size, self.i_size= self.w['ur'].shape
            self.o_size, _= self.w['wo'].shape
        else:
            self.i_size = i_size
            self.h_size = h_size
            self.o_size = o_size
            self.w={}
            self.b={}
