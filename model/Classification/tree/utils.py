import numpy as np

def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)

    return -np.sum(
        [p * np.log2(p) for p in ps if p > 0])