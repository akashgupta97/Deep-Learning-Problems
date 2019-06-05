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
    # RNN training parameter
    batch = 100
    miter = 20
    epoch0 = epoch = 3

    print "training start."
    while epoch > 0:
        itr = 0
        while itr < miter:
            deltaw = {}
            deltab= {}
            err = 0

            # mini_batch foramtion
            mini_batch = [samples[np.random.randint(0, len(samples))] for i in range(batch)]

            # mini_batch training
            while mini_batch:
                x,y = mini_batch.pop()
                model.forward_pass(x)
                dw, db, e = model.backward_pass(y)
                for j in dw:
                    if j in deltaw:
                        deltaw[j]+=dw[j]
                    else:
                        deltaw[j]=dw[j]
                for j in db:
                    if j in deltab:
                        deltab[j]+=db[j]
                    else:
                        deltab[j]=db[j]
                err += e
