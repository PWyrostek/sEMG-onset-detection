import numpy as np


def onset_hidden_factor(data, M, W, h):
    W_2 = round(W / 2)
    initial_var = np.var(data[0:W_2 * 2 + 1])

    variances = [0] * len(data)

    for n in range(W_2, len(data) - W_2):
        variances[n] = np.var(data[n - W_2: n + W_2])
    high_var = max(variances)
    threshold_var = initial_var + (high_var - initial_var) * h

    for n in range(W_2, len(data) - W_2):
        current_var = variances[n]
        if current_var > threshold_var:
            return n
    return -1
