import numpy as np


def onset_solnik(data, W, h, duration):
    def count_current_energy(k):
        return abs(data[k] ** 2 - data[k - 1] * data[k + 1])

    initial_mean = np.mean(abs(data[0:W]))
    initial_std = np.std(data[0:W])
    for n in range(1, len(data) - duration):
        count_subsequent = 0
        for m in range(n, n + duration):
            current_energy = count_current_energy(m)
            if current_energy > (initial_mean + h * initial_std):
                count_subsequent += 1
            else:
                break
        if count_subsequent >= duration:
            return n
    return -1
