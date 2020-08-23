import numpy as np
from utilities import estimate_theta_0


def onset_AGLRstep(data, h, W, M):
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

    delta = 100
    theta_0 = estimate_theta_0(data, M)
    values = [(count_log_likelihood_ratio_step(data, k - W, k, theta_0), k) for k in range(W, len(data))]
    values = [item for item in values if item[0] >= h]
    t_a = values[0][1]
    log_likelihood_list = [(count_log_likelihood_ratio_step(data, j, t_a + delta, theta_0), j) for j in
                           range(W, t_a + 1)]
    return max(log_likelihood_list)[1]


def function_test_AGLRs(data, results, begin, end):
    """Function evaluated by every process - can't be an inner function due to pickling issues"""

    def get_diffs(function, data, column):
        result = data[column, 7]
        if result >= 0:
            emg_single_data = data[:, column]
            data_length = len(emg_single_data)
            diffs = []
            for h in range(1, 20):
                print("{0}".format(h))
                for W in range(5, 20):
                    print("{0} {1}".format(h, W))
                    for M in range(10, 16):
                        try:
                            diffs.append((abs((function(emg_single_data, h * 10, W * 10, M * 15) - result) ** 2),
                                          (h * 10, W * 10, M * 15)))
                        except:
                            diffs.append(
                                (data_length ** 2, (h * 10, W * 10, M * 15)))
            return diffs
        else:
            return None

    diffs_list = []
    for i in range(begin, end + 1):
        for j in range(0, 6):
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ {0} {1}".format(i, j))
            result = get_diffs(onset_AGLRstep, data['emg{0}'.format(i)], j)
            if result != None:
                diffs_list.append(result)
    results.append(diffs_list)


def onset_AGLRramp(data):
    def estimate_theta_1_ramp(data, j, k, theta_0, tau):
        sum_upper = 0
        sum_lower = 0
        for i in range(j, k + 1):
            sum_upper += (data[i] ** 2 - theta_0)
            sum_lower += count_unit_ramp(i, j, tau)
        return sum_upper / sum_lower

    def count_log_likelihood_ratio_ramp(data, j, k, theta_0, tau):
        sum = 0
        for i in range(j, k + 1):
            theta_1 = estimate_theta_1_ramp(data, i, j, theta_0, tau)
            sum += (1 / theta_0 - 1 / (theta_1 + theta_0)) * (data[i] ** 2) + (np.log(theta_0 / (theta_1 + theta_0)))
        return sum / 2

    def count_unit_ramp(k, t_0, tau):
        """Unit ramp for AGLRramp: https://imgur.com/Y70WfDn"""
        if k < t_0:
            return 0
        elif k > (t_0 + tau):
            return 1
        else:
            return (k - t_0) / tau

    W = 25
    h = 10
    delta = 100
    M = 200
    tau = 10  # duration of the ramp
    theta_0 = estimate_theta_0(data, M)
    values = [(k, count_log_likelihood_ratio_ramp(data, k - W + 1, k, theta_0, tau)) for k in range(W, len(data))]
    print(values)
