import numpy as np


def onset_londral(data, W, h, duration):
    def test_function(k):
        first_sum = 0
        second_sum = 0
        for i in range(k - W, k):
            first_sum += (data[i] ** 2)
            second_sum += data[i]
        second_sum = (second_sum ** 2) / W
        return (first_sum - second_sum) / W

    initial_mean = np.mean(abs(data[0:W]))
    initial_std = np.std(data[0:W])
    test_values = []
    for i in range(0, W - 1):
        test_values.append(test_function(i))

    for n in range(0, len(data) - W):
        count_subsequent = 0
        test_values.append(test_function(n + W - 1))
        for m in range(n, n + W):
            current_FVal = test_values[m]
            if current_FVal > (initial_mean + h * initial_std):
                count_subsequent += 1
            else:
                break
        if count_subsequent >= duration:
            return n
    return -1
