import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from sklearn.decomposition import PCA
import math
import multiprocessing

np.seterr(divide='ignore')
BASE_FREQUENCY = 1000
THREADS_AMOUNT = 1


def main():
    data_column = 0
    mat_data = sio.loadmat('database.mat')
    emg_data = mat_data['emg1']
    torque_data = emg_data[:, 6]
    emg_single_data = emg_data[:, data_column]

    result = emg_data[data_column, 7]

    print("SHOULD BE {0}".format(result))
    make_plot(emg_single_data, torque_data, result, onset_two_step_alg(emg_single_data, 150, 1, 0.03, 10, 50, 200))


def split(list, n):
    """Splits list into n parts"""
    k, m = divmod(len(list), n)
    return (list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def make_plot(emg_data, torque_data, expected, found_onset=0, found_right_side=5000):
    """Creates a plot presenting voltage and torque"""
    fig, axs = plt.subplots(2)
    plt.style.use('seaborn-whitegrid')
    axs[0].plot(emg_data, linewidth=1)
    axs[1].plot(torque_data, linewidth=1, color="red")
    axs[0].axvline(x=found_onset, color='tab:orange', alpha=0.5, linewidth=4)
    axs[0].axvline(x=expected, color='tab:green', alpha=0.5, linewidth=4)
    axs[0].axvline(x=found_right_side, color='tab:pink', alpha=0.5, linewidth=4)
    fig = plt.gcf()
    fig.set_size_inches(19.2, 10.8)
    plt.setp(axs, xticks=[i for i in range(0, len(emg_data), 100)])
    plt.show()


def find_optimal_params(data, function):
    diffs_list = []
    threads = []
    slices = (list(split(range(2, 21), THREADS_AMOUNT)))

    manager = multiprocessing.Manager()
    results = manager.list()
    for i in range(THREADS_AMOUNT):
        t = multiprocessing.Process(target=function, args=(
            data, results, slices[i][0], slices[i][-1] + 1 if len(slices[i]) == 1 else slices[i][-1],))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    for result in results:
        diffs_list.extend(result)

    diffs_sum = []

    if function == function_test_komi:
        length = (min([len(diffs_list[i]) for i in range(0, len(diffs_list))]))
    else:
        length = len(diffs_list[0])

    for i in range(0, length):
        diffs_sum.append(
            (sum(row[i][0] for row in diffs_list), diffs_list[0][i][1]))
    # print(diffs_sum)
    best_param = (min(diffs_sum)[1])
    return best_param


def function_test_twostep(data, results, begin, end):
    """Function evaluated by every process - can't be an inner function due to pickling issues"""

    def get_diffs(function, data, column):
        result = data[column, 7]
        if result >= 0:
            emg_single_data = data[:, column]
            diffs = []
            for W_1 in range(10, 21):
                print(W_1)
                for k_1 in range(1, 4):
                    print("@@@ {0}".format(k_1))
                    for d_1 in range(1, 4):
                        print("###### {0}".format(d_1))
                        for h_2 in range(2, 20):
                            print("%%%%%%%%% {0}".format(h_2))
                            for W_2 in range(1, 11):
                                print("&&&&&&&&&&&&&& {0}".format(W_2))
                                for M_2 in range(15, 21):
                                    diffs.append(
                                        (abs((function(emg_single_data, W_1 * 10, k_1, d_1 / 100, h_2, W_2 * 5,
                                                       M_2 * 10) - result) ** 2), (W_1 * 10, k_1, d_1 / 100, h_2,
                                                                                   W_2 * 5,
                                                                                   M_2 * 10)))

            return diffs
        else:
            return None

    diffs_list = []
    for i in range(begin, end):
        for j in range(0, 6):
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ {0} {1}".format(i, j))
            result = get_diffs(onset_two_step_alg, data['emg{0}'.format(i)], j)
            if result != None:
                diffs_list.append(result)
    results.append(diffs_list)


def function_test_AGLRs(data, results, begin, end):
    """Function evaluated by every process - can't be an inner function due to pickling issues"""

    def get_diffs(function, data, column):
        result = data[column, 7]
        if result >= 0:
            emg_single_data = data[:, column]
            diffs = []
            for h in range(2, 20):
                print(h)
                for W in range(1, 11):
                    for M in range(15, 21):
                        diffs.append(
                            (abs((function(emg_single_data, h, W * 5, M * 10) - result) ** 2), (h, W * 5, M * 10)))
            return diffs
        else:
            return None

    diffs_list = []
    for i in range(begin, end):
        for j in range(0, 6):
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ {0} {1}".format(i, j))
            result = get_diffs(onset_AGLRstep, data['emg{0}'.format(i)], j)
            if result != None:
                diffs_list.append(result)
    results.append(diffs_list)


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
    for i in range(begin, end):
        for j in range(0, 6):
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ {0} {1}".format(i, j))
            result = get_diffs(onset_hodges_bui, data['emg{0}'.format(i)], j)
            if result != None:
                diffs_list.append(result)
    results.append(diffs_list)


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
    for i in range(begin, end):
        for j in range(0, 6):
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ {0} {1}".format(i, j))
            result = get_diffs(onset_teager_kaiser, data['emg{0}'.format(i)], j)
            if result != None:
                diffs_list.append(result)
    results.append(diffs_list)


def function_test_komi(data, result, begin, end):
    """Function evaluated by every process - can't be an inner function due to pickling issues"""

    def get_diffs(function, data, column):
        result = data[column, 7]
        emg_single_data = data[:, column]
        diffs = [(abs(function(emg_single_data, i / 1000) - result), (i / 1000)) for i in
                 range(0, math.floor(max(emg_single_data) * 1000))]
        return diffs

    diffs_list = []
    for i in range(begin, end):
        for j in range(0, 6):
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ {0} {1}".format(i, j))
            diffs_list.append(get_diffs(onset_komi, data['emg{0}'.format(i)], j))
    result.append(diffs_list)


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

    # h = 10
    # W = 25
    delta = 100
    # M = 200
    theta_0 = estimate_theta_0(data[left:right], M)
    values = [(k, count_log_likelihood_ratio_step(data, k - W, k, theta_0)) for k in range(W + left, right)]
    values = [item for item in values if item[1] >= h]
    # print(values)
    t_a = min(values)[0]
    # print(t_a)
    log_likelihood_list = [(count_log_likelihood_ratio_step(data, j, t_a + delta, theta_0), j) for j in
                           range(W + left, t_a + 1)]
    return max(log_likelihood_list)[1]


def onset_two_step_alg(data, W_1, k_1, d_1, h_2, W_2, M_2):
    left, right = onset_sign_changes(data, W_1, k_1, d_1)  # 200,1,0.01
    print(left, right)
    result = onset_AGLRstep_two_step(data, h_2, W_2, M_2, left, right)  # 20,15,20
    # print(result)
    return result


def onset_sign_changes(data, W, k, d):
    def find_left_side():
        for i in range(0, len(data)):
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
            endWindow = 0
            if i + W > (len(data) - 1):
                endWindow = len(data) - 1
            else:
                endWindow = i + W

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

    # W = 25
    # h = 10
    delta = 100
    # M = 200
    theta_0 = estimate_theta_0(data, M)
    values = [(k, count_log_likelihood_ratio_step(data, k - W, k, theta_0)) for k in range(W, len(data))]
    values = [item for item in values if item[1] >= h]
    t_a = min(values)[0]
    log_likelihood_list = [(count_log_likelihood_ratio_step(data, j, t_a + delta, theta_0), j) for j in
                           range(W, t_a + 1)]
    return max(log_likelihood_list)[1]


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


def onset_teager_kaiser(data, W, k):
    var = np.var(data[0:W])
    for i in range(0, len(data)):
        if np.var(data[i:i + W]) > var * k:
            return i + W / 2


def onset_komi(data, h):
    data = abs(data)
    # h=0.03
    for i in range(0, len(data)):
        if data[i] > h:
            return i


def estimate_theta_0(data, M):
    sum = 0
    for i in range(0, M):
        sum += data[i] ** 2
    return sum / M


def onset_AGLRramp(data):
    def estimate_theta_1_ramp(data, j, k, theta_0, tau):
        sum_upper = 0
        sum_lower = 0
        for i in range(j, k + 1):
            sum_upper += (data[i] ** 2 - theta_0)
            sum_lower += count_unit_ramp(i, j, tau)  # i+1?
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

    # data=abs(data)
    W = 25
    h = 10
    delta = 100
    M = 200
    tau = 10  # duration of the ramp
    theta_0 = estimate_theta_0(data, M)
    values = [(k, count_log_likelihood_ratio_ramp(data, k - W + 1, k, theta_0, tau)) for k in range(W, len(data))]
    print(values)


def butterworth_filter(data, cutoff_frequency):
    """Applies butterworth filter with given cutoff frequency to the data"""
    w = cutoff_frequency / (BASE_FREQUENCY / 2)
    b, a = signal.butter(6, w, btype='lowpass', analog=False)
    data = signal.filtfilt(b, a, data)
    return data


def whitening(data):
    """TODO: Whitening filter"""
    pca = PCA(whiten=True)
    whitened = pca.fit_transform(data.reshape(-1, 1))
    return whitened


if __name__ == "__main__":
    main()
