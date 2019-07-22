import numpy as np, pickle

def sample_formation(text, seq_length, map_vect):
    samples = []
    t_size = len(text)
    for i in range(0, t_size - seq_length - 1):
        x = [map_vect[j] for j in text[i: i + seq_length]]
        y = [map_vect[j] for j in text[i + 1: i + seq_length + 1]]
        samples.append((x, y))
    return samples

def train(fname, rnn, map_vect):
    text = open(fname,'r').read()
    chars = list(set(text))
    v_size, t_size = len(chars), len(text)

    # recurrent NN initalization
    model = rnn(v_size, 250, v_size, optimize='rmsprop')

    # sample generation
    seq_length = 25
    samples = sample_formation(text, seq_length, map_vect)
