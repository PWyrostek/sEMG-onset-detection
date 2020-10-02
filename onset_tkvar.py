import numpy as np


def onset_TKVar(data, W, g):
    var = np.var(data[0:W])
    h = var * g

    for i in range(0, len(data) - W):
        if np.var(data[i:i + W]) > h:
            return i + W / 2
    return -1
