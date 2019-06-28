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
