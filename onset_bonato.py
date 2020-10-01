import numpy as np


def onset_bonato(data, h, duration, num_of_all_active, pool_size):
    def test_function(k):
        result = (data[k - 1] ** 2 + data[k] ** 2) / var
        return result

    var = np.var(data[0:pool_size])
    for n in range(2, len(data) - pool_size, 2):
        count_all = 0
        count_subsequent = 0
        is_subsequent = True
        for m in range(n, n + pool_size):
            current_FVal = test_function(m)
            if current_FVal > h:
                if is_subsequent:
                    count_subsequent += 1
                count_all += 1
            else:
                is_subsequent = False
        if count_subsequent >= duration and count_all >= num_of_all_active:
            return n
    return -1
