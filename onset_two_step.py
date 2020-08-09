import numpy as np
from utilities import estimate_theta_0

def function_test_sign_changes(data, results, begin, end):
    """Function evaluated by every process - can't be an inner function due to pickling issues"""

    def get_diffs(function, data, column):
        result = data[column, 7]
        if result >= 0:
            emg_single_data = data[:, column]
            data_length = len(emg_single_data)
            diffs = []
            for W in range(15, 31):
                print(W)
                for k in range(1, 2):
                    print("@@@ {0} {1}".format(W, k))
                    for d in range(1, 9):
                        print("###### {0} {1} {2}".format(W, k, d))
                        value = function(emg_single_data, W * 10, k, d / 400)[0]
                        if value is not None and value <= result:
                            diffs.append((abs((value - result) ** 2), (W * 10, k, d / 400)))
                        else:
                            diffs.append((data_length ** 3, (W * 10, k, d / 400)))
            return diffs
        else:
            return None

    diffs_list = []
    for i in range(begin, end + 1):
        for j in range(0, 6):
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ {0} {1}".format(i, j))
            result = get_diffs(onset_sign_changes, data['emg{0}'.format(i)], j)
            if result != None:
                diffs_list.append(result)
    results.append(diffs_list)


def function_test_AGLRs_after_step(data, results, begin, end, sign_changes_params):
    """Function evaluated by every process - can't be an inner function due to pickling issues"""

    def get_diffs(function, data, column):
        result = data[column, 7]
        if result >= 0:
            emg_single_data = data[:, column]
            data_length = len(emg_single_data)
            diffs = []
            for h in range(1, 20):
                print('{0}'.format(h))
                for W in range(10, 20):
                    print('{0} {1}'.format(h, W))
                    for M in range(10, 16):
                        try:
                            value = function(emg_single_data, *sign_changes_params, h * 10, W * 5, M * 15)
                            diffs.append(
                                (abs((value - result) ** 2), (h * 10, W * 5, M * 15)))
                        except:
                            diffs.append(
                                (data_length ** 2, (h * 10, W * 5, M * 15)))

            return diffs
        else:
            return None

    diffs_list = []
    print("{0} {1}".format(begin,end))
    for i in range(begin, end+1):
        for j in range(0, 6):
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ {0} {1}".format(i, j))
            result = get_diffs(onset_two_step_alg, data['emg{0}'.format(i)], j)
            if result != None:
                diffs_list.append(result)
    results.append(diffs_list)


def function_test_twostep(data, results, begin, end):
    """Function evaluated by every process - can't be an inner function due to pickling issues"""

    def get_diffs(function, data, column):
        result = data[column, 7]
        if result >= 0:
            emg_single_data = data[:, column]
            data_length = len(emg_single_data)
            diffs = []
            for W_1 in range(10, 21):
                print(W_1)
                for k_1 in range(1, 2):
                    print("@@@ {0} {1}".format(W_1, k_1))
                    for d_1 in range(1, 4):
                        print("###### {0} {1} {2}".format(W_1, k_1, d_1))
                        for h_2 in range(2, 20):
                            print("%%%%%%%%% {0} {1} {2} {3}".format(W_1, k_1, d_1, h_2))
                            for W_2 in range(5, 16):
                                print("&&&&&&&&&&&&&& {0} {1} {2} {3} {4}".format(W_1, k_1, d_1, h_2, W_2))
                                for M_2 in range(15, 21):
                                    try:
                                        value = function(emg_single_data, W_1 * 10, k_1, d_1 / 100, h_2, W_2 * 5,
                                                         M_2 * 10)
                                        diffs.append((abs((value - result) ** 2), (W_1 * 10, k_1, d_1 / 100, h_2,
                                                                                   W_2 * 5,
                                                                                   M_2 * 10)))
                                    except:
                                        diffs.append(
                                            (data_length ** 2, (W_1 * 10, k_1, d_1 / 100, h_2,
                                                                W_2 * 5,
                                                                M_2 * 10)))

            return diffs
        else:
            return None

    diffs_list = []
    for i in range(begin, end + 1):
        for j in range(0, 6):
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ {0} {1}".format(i, j))
            result = get_diffs(onset_two_step_alg, data['emg{0}'.format(i)], j)
            if result != None:
                diffs_list.append(result)
    results.append(diffs_list)


def onset_AGLRstep_two_step(data, h, W, M, left, right):
    def estimate_theta_1_step(data, j, k):
        sum = 0
        for i in range(j, k + 1):
            sum += data[i] ** 2
        return sum / (k - j + 1)

    def count_log_likelihood_ratio_step(data, j, k, theta_0):
        value = (k - j + 1) / 2
        value *= ((estimate_theta_1_step(data, j, k) / theta_0) - np.log(
            estimate_theta_1_step(data, j, k) / theta_0) - 1)
        return value

    # W = 25
    delta = 100
    # M = 200
    theta_0 = estimate_theta_0(data, M)
    values = [(k, count_log_likelihood_ratio_step(data, k - W, k, theta_0)) for k in range(W + left, right)]
    # print(values)
    values = [item for item in values if item[1] >= h]
    # print(values)
    t_a = min(values)[0]
    # print(t_a)
    log_likelihood_list = [(count_log_likelihood_ratio_step(data, j, t_a + delta, theta_0), j) for j in
                           range(W + left, t_a + 1)]
    # print(log_likelihood_list)
    return max(log_likelihood_list)[1]


def onset_two_step_alg(data, W_1, k_1, d_1, h_2, W_2, M_2):
    left, right = onset_sign_changes(data, W_1, k_1, d_1)  # 200,1,0.01
    result = onset_AGLRstep_two_step(data, h_2, W_2, M_2, left, right)  # 20,15,20
    # print(result)
    return result


def onset_sign_changes(data, W, k, d):
    def find_left_side():
        for i in range(0, len(variability) - W // 4):
            endWindow = 0
            if i + W > (len(data) - 1):
                endWindow = len(data) - 1
            else:
                endWindow = i + W

            if variability[i] > 1 and variability[i + W // 4] > 1 and data[i] > 0.008 and max(
                    variability[i:endWindow]) >= 5 and max(variability[i:endWindow]) >= variability[i] * 1.5 or \
                    variability[i] == max(variability):
                return i

    def find_right_side():
        for i in range(0, len(data)):
            if variability[i] == max(variability):
                return i

    def diff(list):
        return [list[i + 1] - list[i] for i in range(0, len(list) - 1)]

    # W = 200
    # k = 1
    signs = [1 if single_data >= 0 else -1 for single_data in data]
    # d = 0.01
    mul = 1.5
    h = (max(np.abs(diff(data[0:int(W * mul)]))) + d) * k
    data = abs(data)
    variability = []
    for i in range(W // 2, len(data) - W // 2):
        points = 0
        for j in range(i - W // 2 + 1, i + W // 2 - 1):
            if signs[j] == signs[j + 1] and (data[j] - data[j + 1]) > h:
                points += 1
        variability.append(points)

    # print([(i, variability[i], data[i]) for i in range(0,len(variability))])
    return (find_left_side(), find_right_side())