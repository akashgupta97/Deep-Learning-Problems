import numpy as np
from util import softmax, sigmoid, dsigmoid, adam, rmsprop

import pickle

class vrnn:

    def __init__(self, i_size, h_size, o_size, optimize='rmsprop', wb=None):

        self.i_size = i_size
        self.h_size = h_size
        self.o_size = o_size
        self.optimize = optimize

        if wb:
            self.w, self.b = self.load_model(wb)
        else:
            self.w={}
            self.b={}

            # input to hidden weights
            self.w['ih'] = np.random.normal(0, 0.01, (h_size, i_size))
            self.b['ih'] = np.zeros((h_size, 1))

            # prev hidden to hidden weights
            self.w['ph'] = np.random.normal(0, 0.01, (h_size, h_size))
            self.b['ph'] = np.zeros((h_size, 1))

            # hidden to output weights
            self.w['ho'] = np.random.normal(0,0.01,(o_size, h_size))
            self.b['ho'] = np.zeros((o_size, 1))

        if optimize == 'rmsprop' or optimize == 'adam':
            self.m={}
            self.m['ih'] = np.zeros((h_size, i_size))
            self.m['ph'] = np.zeros((h_size, h_size))
            self.m['ho'] = np.zeros((o_size, h_size))

        if optimize == 'adam':
            self.v={}
            self.v['ih'] = np.zeros((h_size, i_size))
            self.v['ph'] = np.zeros((h_size, h_size))
            self.v['ho'] = np.zeros((o_size, h_size))
            self.weight_update = adam
