import numpy as np


def onset_teager_kaiser(data, W, k):
    var = np.var(data[0:W])
    for i in range(0, len(data) - W):
        if np.var(data[i:i + W]) > var * k:
            return i + W / 2
    return -1
