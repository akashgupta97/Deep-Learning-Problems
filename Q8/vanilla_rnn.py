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

        elif optimize == 'rmsprop':
            self.weight_update = rmsprop

    def forward_pass(self, inputs):

        self.inputs = inputs
        self.n_inp = len(inputs)
        self.o = []; self.h = {}
        self.vh = []; self.vo = []
        self.h[-1] = np.zeros((self.h_size, 1))
        for i in range(self.n_inp):
            # calculation for hidden activation
            self.vh.append(np.dot(self.w['ih'], inputs[i]) + np.dot(self.w['ph'], self.h[i - 1]) + self.b['ih'])
            self.h[i] = (sigmoid(self.vh[i]))

            # calculation for output activation
            self.vo.append(np.dot(self.w['ho'], self.h[i]) + self.b['ho'])
            self.o.append(softmax(self.vo[i]))
        return self.o

    def backward_pass(self, t):
        # error calculation
        e = self.error(t)

        # dw variables
        dw={}
        db= {}
        dw['ih'] = np.zeros((self.h_size, self.i_size))
        db['ih'] = np.zeros((self.h_size, 1))
