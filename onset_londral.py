import numpy as np


def onset_londral(data, W, h, duration):
    def test_function(k):
        first_sum = 0
        second_sum = 0
        for i in range(k - duration, k):
            first_sum += (data[i] ** 2)
            second_sum += data[i]
        second_sum = (second_sum ** 2) / duration
        return (first_sum - second_sum) / duration

    initial_mean = np.mean(abs(data[0:W]))
    initial_std = np.std(data[0:W])
    for n in range(W, len(data)):
        count_subsequent = 0
        for m in range(n, n + duration):
            current_FVal = test_function(m)
            if current_FVal > (initial_mean + h * initial_std):
                count_subsequent += 1
            else:
                break
        if count_subsequent >= duration:
            return n
    return -1
