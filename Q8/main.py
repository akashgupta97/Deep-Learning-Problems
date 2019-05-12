from char_pridect import predict_words, get_top_words
import numpy as np, sys, os
from GRU import gru
from vanilla_rnn import vrnn
from rnn_interface import train, test

def usage():
    print "Usage : python main.py [options]"
    print "options:"
    print "\t -rnn vrnn or gru: define type recurrent model use (by default 'gru')"
    print "\t -text file name: input file name use for trainning model for character prediction"
    print "\t -npredict number of pridection: given npredict number of words predicted"
    exit()
