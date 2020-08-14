import numpy as np


def onset_hodges_bui(data, h, W, M):
    def count_y(k):
        sum = 0
        for i in range(k - W, k + 1):
            sum += data[i]
        return sum / W

    def test_function(k):
        return (1 / std) * (count_y(k) - mean)

    data = abs(data)
    # h = 2.5
    # W = 50
    # M = 200
    f_c = 50
    std = np.std(data[0:M])
    mean = np.mean(data[0:M])
    values = [(k, test_function(k)) for k in range(W, len(data))]
    values = [item for item in values if item[1] >= h]

    return values[0][0] - W


def function_test_hodges(data, results, begin, end):
    """Function evaluated by every process - can't be an inner function due to pickling issues"""

    def get_diffs(function, data, column):
        result = data[column, 7]
        if result >= 0:
            emg_single_data = data[:, column]
            diffs = []
            for h in range(20, 41):
                print(h)
                for W in range(1, 11):
                    for M in range(10, 20):
                        diffs.append((abs((function(emg_single_data, h / 10, W * 5, M * 10) - result) ** 2), (h / 10,
                                                                                                              W * 5,
                                                                                                              M * 10)))
            return diffs
        else:
            return None

    diffs_list = []
    for i in range(begin, end + 1):
        for j in range(0, 6):
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ {0} {1}".format(i, j))
            result = get_diffs(onset_hodges_bui, data['emg{0}'.format(i)], j)
            if result != None:
                diffs_list.append(result)
    results.append(diffs_list)
