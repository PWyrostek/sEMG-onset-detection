import scipy.io as sio
import matplotlib.pyplot as plt
import multiprocessing
from onset_komi import *
from onset_teager_kaiser import *
from onset_aglr import *
from onset_hodges_bui import *
from onset_two_step import *
import skopt

THREADS_AMOUNT = 3

DATABASE_NAME = 'database_extended.mat'
DATABASE_TABLE = 'emg1'
DATA_COLUMN = 0


def main():
    mat_data = sio.loadmat(DATABASE_NAME)
    emg_data = mat_data[DATABASE_TABLE]
    torque_data = emg_data[:, 6]
    emg_single_data = emg_data[:, DATA_COLUMN]

    # 230,1,0.005
    # (160, 1, 0.0025)
    # (290, 1, 0.01)
    # 4, 75, 190
    # (7, 110, 160)
    # (1, 160, 150)
    # (3, 120, 180)
    # (190, 55, 150)
    # [x for x in range(1,21) if x not in [3, 8, 11, 14, 19]]
    prepare_results(mat_data, [x for x in range(1, 30) if x not in range(13, 21)])
    # make_histogram(errors)
    # result = emg_data[DATA_COLUMN, 7]
    # found_onset = onset_two_step_alg(emg_single_data, 160, 1, 0.0025, 1, 160, 150)
    # make_plot(emg_single_data, torque_data, result, found_onset)
    # params = find_optimal_params(mat_data, function_test_AGLRs_after_step, [3, 8, 11, 14, 19], (290, 1, 0.01))
    # print(params)
    # params = find_optimal_params(mat_data, function_test_sign_changes, [3, 8, 11, 14, 19])
    # print(params)
    # params = find_optimal_params(mat_data, function_test_AGLRs, [3, 8, 11, 14, 19])
    # print(params)


def prepare_results(database, data_range):
    results = []
    onsets = []
    errors = []
    for j in data_range:
        emg_data = database["emg{0}".format(j)]
        print("@@@@@@@@@@@@@@@@ {0}".format(j))
        for i in range(0, 6):
            emg_single_data = emg_data[:, i]
            result = emg_data[i, 7]
            if result > 0:
                try:
                    print("SHOULD BE {0}".format(result))
                    found_onset = onset_two_step_alg(emg_single_data, 290, 1, 0.01, 190, 55, 150)
                    print(found_onset)
                    results.append(result)
                    onsets.append(found_onset)
                    errors.append((found_onset - result))
                    filename = "emg{0}-{1}".format(j, i)
                    # make_plot(emg_single_data, emg_data[:, 6], filename, result, found_onset)
                except:
                    print("error")
    print(results)
    print(onsets)
    print(errors)
    print(np.mean(np.abs(errors)))
    make_histogram(errors)


def split(list, n):
    k, m = divmod(len(list), n)
    return (list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def make_histogram(errors_data):
    fig, ax = plt.subplots()
    counts, bins, patches = plt.hist(errors_data, density=True, bins=20, histtype='bar')
    ax.set_xticks(bins)
    plt.xticks(bins, rotation='vertical')
    plt.ylabel('Probability')
    plt.xlabel('Error [ms]')
    fig.tight_layout()
    plt.show()


def make_plot(emg_data, torque_data, filename, expected, found_onset=0):
    """Creates a plot presenting voltage and torque"""
    fig, axs = plt.subplots(2)
    plt.style.use('seaborn-whitegrid')
    axs[0].plot(emg_data, linewidth=1)
    axs[0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axs[0].set_title("EMG Signal")
    axs[0].set_ylabel("EMG = [mV]")
    axs[0].set_xlabel("t = [ms]")
    axs[1].plot(torque_data, linewidth=1, color="red")
    axs[0].axvline(x=expected, color='tab:green', alpha=0.5, linewidth=2)
    axs[0].plot(found_onset, 0, 'ro', color='tab:orange', linewidth=5)
    fig = plt.gcf()
    fig.set_size_inches(30, 10)
    plt.setp(axs, xticks=[i for i in range(0, len(emg_data) + 100, 100)])
    plt.savefig('./plots/{0}.png'.format(filename))
    plt.show()


def find_optimal_params(data, function, data_range, sign_changes_params=()):
    # data_range = range(9, 19)
    diffs_list = []
    threads = []
    slices = (list(split(data_range, THREADS_AMOUNT)))

    manager = multiprocessing.Manager()
    results = manager.list()
    for i in range(THREADS_AMOUNT):
        args = (data, results, slices[i][0], slices[i][-1])
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
