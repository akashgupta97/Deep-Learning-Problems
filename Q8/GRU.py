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

            # reset weights
            self.w['ur'] = np.random.normal(0,0.01,(h_size, i_size))
            self.b['r'] = np.zeros((h_size, 1))
            self.w['wr'] = np.random.normal(0,0.01,(h_size, h_size))

            # update weights
            self.w['uz'] = np.random.normal(0,0.01,(h_size, i_size))
            self.b['z'] = np.zeros((h_size, 1))
            self.w['wz'] = np.random.normal(0,0.01,(h_size, h_size))

            # _h weights
            self.w['u_h'] = np.random.normal(0,0.01,(h_size, i_size))
            self.b['_h'] = np.zeros((h_size, 1))
            self.w['w_h'] = np.random.normal(0,0.01,(h_size, h_size))

            # out weight
            self.w['wo'] = np.random.normal(0,0.01,(o_size, h_size))
            self.b['o'] = np.zeros((o_size, 1))
