import numpy as np


def onset_londral(data, W, h, duration):
    initial_mean = np.mean(abs(data[0:W]))
    initial_std = np.std(data[0:W])
    for n in range(W, len(data)):
        count_subsequent = 0
        for m in range(n, n + duration):
            current_FVal = np.var(data[m:m + duration])
            if current_FVal > (initial_mean + h * initial_std):
                count_subsequent += 1
            else:
                break
        if count_subsequent >= duration:
            return n
    return -1
