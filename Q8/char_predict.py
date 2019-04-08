import numpy as np

def predict_chars(text, model, map_vect, i2c_map, n=2):
    x = [map_vect[i] for i in text]
    out = model.forward_pass(x)
    pred = np.argsort(out[-1], axis=0)[-n:,0].tolist()
    return {i2c_map[i]:out[-1][i] for i in pred}
