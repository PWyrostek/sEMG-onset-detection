import numpy as np
from utilities import signal_energy


def onset_TKVar(data, W, g):
    data_energy = signal_energy(data)
    var = np.var(data_energy[0:W])
    h = var * g

    for i in range(0, len(data_energy) - W):
        if np.var(data_energy[i:i + W]) > h:
            return i + W / 2
    return -1
