import numpy as np


def onset_silva(data, w_1, w_2, h):
    beginning = max(w_1, w_2)
    for n in range(beginning, len(data)):
        current_FVal = np.mean(abs(data[n - w_1 + 1: n]))
        current_adaptive_threshold = np.mean(abs(data[n - w_2 + 1: n]))
        if current_FVal >= current_adaptive_threshold and current_FVal >= h:
            return n
    return -1
