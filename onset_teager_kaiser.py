import numpy as np


def onset_teager_kaiser(data, W, k):
    var = np.var(data[0:W])
    for i in range(0, len(data)):
        if np.var(data[i:i + W]) > var * k:
            return i + W / 2
    return -1


def function_test_teager(data, results, begin, end):
    """Function evaluated by every process - can't be an inner function due to pickling issues"""

    def get_diffs(function, data, column):
        result = data[column, 7]
        if result >= 0:
            emg_single_data = data[:, column]
            diffs = []
            for i in range(2, 50):
                for j in range(1, 10):
                    diffs.append((abs((function(emg_single_data, i, j) - result) ** 2), (i, j)))
            return diffs
        else:
            return None

    diffs_list = []
    for i in range(begin, end + 1):
        for j in range(0, 6):
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ {0} {1}".format(i, j))
            result = get_diffs(onset_teager_kaiser, data['emg{0}'.format(i)], j)
            if result != None:
                diffs_list.append(result)
    results.append(diffs_list)
