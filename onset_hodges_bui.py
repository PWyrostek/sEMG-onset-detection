import numpy as np


def onset_hodges_bui(data, W, h):
    initial_mean = np.mean(abs(data[0:W]))
    initial_std = np.std(data[0:W])

    for n in range(W, len(data)):
        current_mean = np.mean(abs(data[n - W:n]))
        current_FVal = current_mean - initial_mean
        if current_FVal > (h * initial_std):
            return n
    return -1
