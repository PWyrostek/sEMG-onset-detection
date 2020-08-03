import scipy.io as sio
import matplotlib.pyplot as plt
import multiprocessing
from onset_komi import *
from onset_teager_kaiser import *
from onset_aglr import *
from onset_hodges_bui import *
from onset_two_step import *

THREADS_AMOUNT = 6

DATABASE_NAME = 'database.mat'
DATABASE_TABLE = 'emg1'
DATA_COLUMN = 0


def main():
    mat_data = sio.loadmat(DATABASE_NAME)
    emg_data = mat_data[DATABASE_TABLE]
    torque_data = emg_data[:, 6]
    emg_single_data = emg_data[:, DATA_COLUMN]

    result = emg_data[DATA_COLUMN, 7]

    print("SHOULD BE {0}".format(result))
    # found_onset = onset_two_step_alg(emg_single_data, 170, 1, 0.01, 2, 25, 200)
    # print(found_onset)
    # make_plot(emg_single_data, torque_data, result, found_onset)
    params = find_optimal_params(mat_data,function_test_AGLRs_after_step, range(9,20), (200, 1, 0.01))
    print(params)


def split(list, n):
    k, m = divmod(len(list), n)
    return (list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def make_plot(emg_data, torque_data, expected, found_onset=0):
    """Creates a plot presenting voltage and torque"""
    fig, axs = plt.subplots(2)
    plt.style.use('seaborn-whitegrid')
    axs[0].plot(emg_data, linewidth=1)
    axs[1].plot(torque_data, linewidth=1, color="red")
    axs[0].axvline(x=found_onset, color='tab:orange', alpha=0.5, linewidth=4)
    axs[0].axvline(x=expected, color='tab:green', alpha=0.5, linewidth=4)
    fig = plt.gcf()
    fig.set_size_inches(19.2, 10.8)
    plt.setp(axs, xticks=[i for i in range(0, len(emg_data), 100)])
    plt.show()


def find_optimal_params(data, function, data_range, sign_changes_params = ()):
    # data_range = range(9, 19)
    diffs_list = []
    threads = []
    slices = (list(split(data_range, THREADS_AMOUNT)))

    manager = multiprocessing.Manager()
    results = manager.list()
    for i in range(THREADS_AMOUNT):
        args = (data, results, slices[i][0], slices[i][-1] + 1 if len(slices[i]) == 1 else slices[i][-1],)
        if function == function_test_AGLRs_after_step:
            args = args + (sign_changes_params,)
        t = multiprocessing.Process(target=function, args=args)
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


if __name__ == "__main__":
    main()
