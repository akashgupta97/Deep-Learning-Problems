import numpy as np

def predict_chars(text, model, map_vect, i2c_map, n=2):
    x = [map_vect[i] for i in text]
    out = model.forward_pass(x)
    pred = np.argsort(out[-1], axis=0)[-n:,0].tolist()
    return {i2c_map[i]:out[-1][i] for i in pred}

def get_top_words(words, n):
    a = [words[i] for i in words]
    a.sort(reverse=True)
    tword = []
    itr=0
    while itr<n and itr<len(a):
        for j in words:
            if words[j] == a[itr]:
                tword.append(j)
        itr += 1
    return tword
