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

        ### Training variable and function
        cost = T.mean(T.nnet.categorical_crossentropy(py_x, thY))
        updates = rmsprop(cost, params=self.params, lr=lr)

        self.train_op = theano.function(
            inputs=[thX, thY, thStartPoints],
            outputs=[cost, prediction, py_x],
            updates=updates,
        )
        ###

        ### iterating over input values
        n_batches = len(X) // batch_sz
        for i in range(epochs):
            # t0 = datetime.datetime.now()
            X, Y = shuffle(X, Y)
            tn_correct = 0
            tn_total = 0
            cost = 0
            for j in range(n_batches):
                n_correct = 0
                n_total = 0
                sequenceLengths = []
                input_sequence, output_sequence = [], []
                for k in range(j * batch_sz, (j + 1) * batch_sz):
                    # don't always add the end token
                    input_sequence += X[k]
                    output_sequence += Y[k]
                    sequenceLengths.append(len(X[k]))

                startPoints = np.zeros(sum(sequenceLengths), dtype=np.int32)
                last = 0
                for length in sequenceLengths:
                    startPoints[last] = 1
                    last += length
                # try:
                #     input_sequence = np.array(input_sequence, dtype=np.int32)
                #     output_sequence = np.array(output_sequence, dtype=np.int32)
                # except ValueError:
                #     exit()
                c, p, res = self.train_op(input_sequence, output_sequence, startPoints)

                cost += c

                for pj, yj in zip(p, output_sequence):
                    if pj == np.argmax(yj):
                        n_correct += 1
                tn_correct += n_correct
                tn_total += len(output_sequence)
                print("batch: %d/%d" % (j, n_batches), "cost:", c, "accuracy:",(float(n_correct) / len(output_sequence)))
            print("\nepoch: %d/%d cost: %f accuracy: %f\n"%(i,epochs,cost,float(tn_correct/tn_total)))
