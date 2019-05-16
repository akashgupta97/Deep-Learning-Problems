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

def parse_argv(argv):
    rnn = ''; ftext=''; npredict=10
    if len(argv) > 1:
        if argv[1] == '-help' or argv[1] == '--help' or argv[1] == '-h':
            usage()
        if '-rnn' in argv:
            ind=argv.index('-rnn')
            rnn = sys.argv[ind+1]
        if '-text' in argv:
            ind=argv.index('-text')
            ftext = sys.argv[ind+1]
        if '-npredict' in sys.argv:
            ind=sys.argv.index('-npredict')
            npredict = int(sys.argv[ind+1])
        if not os.path.isfile(ftext):
            print 'Enter a correct file path !!'
            exit()

    rnn = vrnn if rnn == 'vrnn' else gru
    fname = ftext if ftext else 'pg.txt'
    return rnn, fname, npredict
