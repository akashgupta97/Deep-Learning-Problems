import numpy as np, pickle

def sample_formation(text, seq_length, map_vect):
    samples = []
    t_size = len(text)
    for i in range(0, t_size - seq_length - 1):
        x = [map_vect[j] for j in text[i: i + seq_length]]
        y = [map_vect[j] for j in text[i + 1: i + seq_length + 1]]
        samples.append((x, y))
    return samples
