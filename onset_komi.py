import math


def onset_komi(data, h=0.03):
    data = abs(data)
    for i in range(0, len(data)):
        if data[i] > h:
            return i
    return -1


def function_test_komi(data, result, begin, end):
    """Function evaluated by every process - can't be an inner function due to pickling issues"""

    def get_diffs(function, data, column):
        result = data[column, 7]
        emg_single_data = data[:, column]
        diffs = [(abs(function(emg_single_data, i / 1000) - result), (i / 1000)) for i in
                 range(0, math.floor(max(emg_single_data) * 1000))]
        return diffs

    diffs_list = []
    for i in range(begin, end + 1):
        for j in range(0, 6):
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ {0} {1}".format(i, j))
            diffs_list.append(get_diffs(onset_komi, data['emg{0}'.format(i)], j))
    result.append(diffs_list)
