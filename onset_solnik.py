import numpy as np
from utilities import signal_energy


def onset_solnik(data, W, h, duration):
    initial_mean = np.mean(abs(data[0:W]))
    initial_std = np.std(data[0:W])
    data_energy = signal_energy(data)
    for n in range(0, len(data_energy) - duration):
        count_subsequent = 0
        for m in range(n, n + duration):
            current_energy = abs(data_energy[m])
            if current_energy > (initial_mean + h * initial_std):
                count_subsequent += 1
            else:
                break
        if count_subsequent >= duration:
            return n
    return -1
