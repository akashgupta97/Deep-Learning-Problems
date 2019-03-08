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

    def fit(self, X, Y, epochs=50, mu=0.9, reg=0., batch_sz=50, lr=0.002):

        ### Initialize X and Y, calculating other variables
        thX = T.imatrix('X')
        thY = T.imatrix('Y')
        thStartPoints = T.ivector('start_points')

        Z = thX
        for ru in self.hidden_layers:
            Z = ru.output(Z, thStartPoints)
        py_x = T.nnet.softmax(Z.dot(self.Wo) + self.bo)
        prediction = T.argmax(py_x, axis=1)
        ###
