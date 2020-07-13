import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from sklearn.decomposition import PCA
import math
import multiprocessing

np.seterr(divide='ignore')
BASE_FREQUENCY = 1000
THREADS_AMOUNT = 4


def main():
    data_column = 0
    mat_data = sio.loadmat('database.mat')
    emg_data = mat_data['emg1']
    torque_data = emg_data[:, 6]
    emg_single_data = emg_data[:, data_column]
    make_plot(emg_single_data, torque_data)
    result = emg_data[data_column, 7]

    # best_komi_param=find_optimal_komi(mat_data)
    # print("Best parameter found: {0}".format(best_komi_param))
    # print("SHOULD BE {0}".format(result))
    # print("Found onset: {0}".format(onset_komi(emg_single_data,best_komi_param)))

    # best_teager_param=find_optimal_teager(mat_data)
    # print("Best parameter found: {0}".format(best_teager_param))
    # print("SHOULD BE {0}".format(result))
    # print("Found onset: {0}".format(onset_teager_kaiser(emg_single_data,best_teager_param[0],best_teager_param[1])))

    # best_hodges_param=find_optimal_hodges(mat_data)
    # print("Best parameter found: {0}".format(best_hodges_param))
    # print("SHOULD BE {0}".format(result))
    # print("Found onset: {0}".format(onset_hodges_bui(emg_single_data,best_hodges_param[0],best_hodges_param[1],best_hodges_param[2])))

    # best_AGLRs_param=find_optimal_AGLRs(mat_data)
    # print("Best parameter found: {0}".format(best_AGLRs_param))
    # print("SHOULD BE {0}".format(result))
    # print("Found onset: {0}".format(onset_AGLRstep(emg_single_data,best_AGLRs_param[0],best_AGLRs_param[1],best_AGLRs_param[2])))


def split(list, n):
    """Splits list into n parts"""
    k, m = divmod(len(list), n)
    return (list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def make_plot(emg_data, torque_data):
    """Creates a plot presenting voltage and torque"""
    fig, axs = plt.subplots(2)
    plt.style.use('seaborn-whitegrid')
    axs[0].plot(emg_data, linewidth=1)
    axs[1].plot(torque_data, linewidth=1, color="red")
    fig = plt.gcf()
    fig.set_size_inches(19.2, 10.8)
    plt.setp(axs, xticks=[i for i in range(0, len(emg_data), 100)])
    plt.show()


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
                            (abs((function(emg_single_data, h, W * 5, M * 10) - result) ** 2), h, W * 5, M * 10))
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


def find_optimal_AGLRs(data):
    diffs_list = []
    threads = []
    slices = (list(split(range(2, 5 + 1), THREADS_AMOUNT)))
    print(slices)
    manager = multiprocessing.Manager()
    results = manager.list()
    for i in range(THREADS_AMOUNT):
        t = multiprocessing.Process(target=function_test_AGLRs, args=(
            data, results, slices[i][0], slices[i][-1] + 1 if len(slices[i]) == 1 else slices[i][-1],))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    for result in results:
        diffs_list.extend(result)

    # print(diffs_list)
    diffs_sum = []
    for i in range(0, len(diffs_list[0])):
        diffs_sum.append(
            (sum(row[i][0] for row in diffs_list), diffs_list[0][i][1], diffs_list[0][i][2], diffs_list[0][i][3]))
    # print(diffs_sum)
    best_param = (min(diffs_sum)[1], min(diffs_sum)[2], min(diffs_sum)[3])
    return best_param


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
                        diffs.append((abs((function(emg_single_data, h / 10, W * 5, M * 10) - result) ** 2), h / 10,
                                      W * 5, M * 10))
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


def find_optimal_hodges(data):
    diffs_list = []
    threads = []
    slices = (list(split(range(2, 5 + 1), THREADS_AMOUNT)))
    # print(slices)
    manager = multiprocessing.Manager()
    results = manager.list()
    for i in range(THREADS_AMOUNT):
        t = multiprocessing.Process(target=function_test_hodges, args=(
            data, results, slices[i][0], slices[i][-1] + 1 if len(slices[i]) == 1 else slices[i][-1],))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    for result in results:
        diffs_list.extend(result)

    diffs_sum = []
    for i in range(0, len(diffs_list[0])):
        diffs_sum.append(
            (sum(row[i][0] for row in diffs_list), diffs_list[0][i][1], diffs_list[0][i][2], diffs_list[0][i][3]))
    best_param = (min(diffs_sum)[1], min(diffs_sum)[2], min(diffs_sum)[3])
    return best_param


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


def function_test_teager(data, results, begin, end):
    """Function evaluated by every process - can't be an inner function due to pickling issues"""

    def get_diffs(function, data, column):
        result = data[column, 7]
        if result >= 0:
            emg_single_data = data[:, column]
            diffs = []
            for i in range(2, 50):
                for j in range(1, 10):
                    diffs.append((abs((function(emg_single_data, i, j) - result) ** 2), i, j))
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


def find_optimal_teager(data):
    diffs_list = []
    threads = []
    slices = (list(split(range(2, 21 + 1), THREADS_AMOUNT)))
    manager = multiprocessing.Manager()
    results = manager.list()
    for i in range(THREADS_AMOUNT):
        t = multiprocessing.Process(target=function_test_teager, args=(data, results, slices[i][0], slices[i][-1],))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    for result in results:
        diffs_list.extend(result)

    diffs_sum = []
    for i in range(0, len(diffs_list[0])):
        diffs_sum.append((sum(row[i][0] for row in diffs_list), diffs_list[0][i][1], diffs_list[0][i][2]))
    best_param = (min(diffs_sum)[1], min(diffs_sum)[2])
    return best_param


def onset_teager_kaiser(data, W, k):
    var = np.var(data[0:W])
    for i in range(0, len(data)):
        if np.var(data[i:i + W]) > var * k:
            return i + W / 2


def function_test_komi(data, result, begin, end):
    """Function evaluated by every process - can't be an inner function due to pickling issues"""

    def get_diffs(function, data, column):
        result = data[column, 7]
        emg_single_data = data[:, column]
        diffs = [(abs(function(emg_single_data, i / 1000) - result), i / 1000) for i in
                 range(0, math.floor(max(emg_single_data) * 1000))]
        return diffs

    diffs_list = []
    for i in range(begin, end):
        for j in range(0, 6):
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ {0} {1}".format(i, j))
            diffs_list.append(get_diffs(onset_komi, data['emg{0}'.format(i)], j))
    result.append(diffs_list)


def find_optimal_komi(data):
    diffs_list = []

    threads = []
    manager = multiprocessing.Manager()
    results = manager.list()

    slices = (list(split(range(2, 21 + 1), THREADS_AMOUNT)))

    for i in range(THREADS_AMOUNT):
        t = multiprocessing.Process(target=function_test_komi, args=(data, results, slices[i][0], slices[i][-1],))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    for result in results:
        diffs_list.extend(result)

    diffs_sum = []
    length = (min([len(diffs_list[i]) for i in range(0, len(diffs_list))]))

    for i in range(0, length):
        diffs_sum.append((sum(row[i][0] for row in diffs_list), diffs_list[0][i][1]))

    best_param = min(diffs_sum)[1]
    return best_param


def onset_komi(data, h):
    data = abs(data)
    # h=0.03
    for i in range(0, len(data)):
        if data[i] > h:
            return i


def onset_sign_changes(data):
    def diff(list):
        return [list[i + 1] - list[i] for i in range(0, len(list) - 1)]

    W = 100
    k = 3
    signs = [1 if single_data >= 0 else -1 for single_data in data]
    d = 0.01
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
    for i in range(0, len(data)):
        endWindow = 0
        if i + W > (len(data) - 1):
            endWindow = len(data) - 1
        else:
            endWindow = i + W

        if variability[i] > 1 and variability[i + W // 4] > 1 and data[i] > 0.008 and max(
                variability[i:endWindow]) >= 5 and max(variability[i:endWindow]) >= variability[i] * 1.5 or variability[
            i] == max(variability):
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
    values = [(k, count_log_likelihood_ratio_ramp(data, k - W, k, theta_0, tau)) for k in range(W, len(data))]
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
