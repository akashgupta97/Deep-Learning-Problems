import theano, theano.tensor as T
theano.config.optimizer='fast_compile'
theano.config.exception_verbosity='high'

from sklearn.utils import shuffle
import numpy as np
import sys, datetime, pickle, os, re

from Recurrent_Unit import GRU
class RNN_Model:
    def __init__(self, I=None, H=None, rnn_unit=GRU, opt=Adam, activation=T.nnet.elu):
        self.D = I
        self.hidden_layer_sizes = H
        self.rnn_unit = rnn_unit
        self.activation = activation
        self.opt = opt
        self.__setstate__()
