import csv
import multiprocessing
import pprint
import time

import matplotlib.pyplot as plt
import neptune
import neptunecontrib.monitoring.optuna as opt_utils
import neptunecontrib.monitoring.skopt as skopt_utils
import optuna
import scipy.io as sio
import skopt
from scipy import stats
from onset_aglr import *
from onset_bonato import onset_bonato
from onset_hidden_factor import onset_hidden_factor
from onset_komi import *
from onset_londral import onset_londral
from onset_silva import onset_silva
from onset_solnik import onset_solnik
from onset_tkvar import onset_TKVar
from onset_two_step import *
import statistics

pp = pprint.PrettyPrinter(indent=4)
THREADS_AMOUNT = 5

DATABASE_NAME = 'database.mat'
DATABASE_TABLE = 'emg1'
DATA_COLUMN = 0


def main():
    def minimize_depreciated():
        def my_func(params):
            h, M = params
            sum = 0
            for j in [3, 4, 8, 11, 14, 19, 25]:
                emg_data = mat_data['emg{0}'.format(j)]
                for i in range(0, 6):
                    emg_single_data = emg_data[:, i]
                    try:
                        # sum += abs(onset_AGLRstep(emg_single_data, h, W, M) - emg_data[i, 7])
                        # sum += abs(onset_sign_changes(emg_single_data, h, 1, M)[0] - emg_data[i, 7])
                        # sum += abs(onset_two_step_alg(emg_single_data, 290, 1, 0.01, h, W, M)- emg_data[i, 7])
                        value = onset_sign_changes(emg_single_data, h, 1, M)[0]
                        result = emg_data[i, 7]
                        if value is not None and value <= result:
                            sum += abs(value - result)
                        else:
                            sum += 5000 ** 2
                    except:
                        sum += 5000 ** 2
            cost = sum
            return cost

        neptune.init(
            api_token='ANONYMOUS',
            project_qualified_name='shared/showroom')

        neptune.create_experiment('minimal_example')
        neptune_callback = skopt_utils.NeptuneCallback()
        bounds = [(100, 400), (0.0025, 0.03)]
        result = skopt.gp_minimize(my_func, bounds, n_calls=500, acq_optimizer="lbfgs", n_jobs=-1,
                                   callback=[neptune_callback])

    def find_minimizing_params():
        def objective_sign_changes_first_step(trial):
            W = trial.suggest_int('W', 100, 400)
            k = trial.suggest_int('k', 1, 3)
            d = trial.suggest_uniform('d', 0.0025, 0.03)
            sum = 0
            for j in [3, 4, 8, 11, 14, 19, 25]:
                emg_data = mat_data['emg{0}'.format(j)]
                for i in range(0, 6):
                    emg_single_data = emg_data[:, i]
                    try:
                        value = onset_sign_changes(emg_single_data, W, k, d)[0]
                        result = emg_data[i, 7]
                        if value is not None and value <= result:
                            sum += abs(value - result)
                        else:
                            sum += abs(value - result)
                            sum += 5000 ** 2
                    except:
                        sum += 2 * (5000 ** 2)
            cost = sum
            return cost

        def objective_sign_changes_standalone(trial):
            W = trial.suggest_int('W', 50, 600)
            k = trial.suggest_int('k', 1, 3)
            d = trial.suggest_uniform('d', 0.0025, 0.03)
            sum = 0
            for j in [3, 4, 8, 11, 14, 19, 25]:
                emg_data = mat_data['emg{0}'.format(j)]
                for i in range(0, 6):
                    emg_single_data = emg_data[:, i]
                    try:
                        value = onset_sign_changes(emg_single_data, W, k, d)[0]
                        result = emg_data[i, 7]
                        sum += abs(value - result)
                    except:
                        sum += 5000 ** 2
            cost = sum
            return cost

        def objective_AGLRs_standalone(trial):
            h = trial.suggest_int('h', 1, 300)
            W = trial.suggest_int('W', 1, 300)
            M = trial.suggest_int('M', 50, 250)
            sum = 0
            for j in [3, 4, 8, 11, 14, 19, 25]:
                emg_data = mat_data['emg{0}'.format(j)]
                for i in range(0, 6):
                    emg_single_data = emg_data[:, i]
                    try:
                        value = onset_AGLRstep(emg_single_data, h, W, M)
                        result = emg_data[i, 7]
                        sum += abs(value - result)
                    except:
                        sum += 5000 ** 2
            cost = sum
            return cost

        def objective_AGLRs_second_step(trial):
            h = trial.suggest_int('h', 1, 300)
            W = trial.suggest_int('W', 10, 100)
            M = trial.suggest_int('M', 50, 250)
            sum = 0
            for j in [3, 4, 8, 11, 14, 19, 25]:
                emg_data = mat_data['emg{0}'.format(j)]
                for i in range(0, 6):
                    emg_single_data = emg_data[:, i]
                    try:
                        value = onset_two_step_alg(emg_single_data, 120, 1, 0.00724023569, h, W, M)
                        result = emg_data[i, 7]
                        sum += abs(value - result)
                    except:
                        sum += 5000 ** 2
            cost = sum
            return cost

        neptune.init(project_qualified_name='pwyrostek/sandbox',
                     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiZjM0ZTRiNjAtNDAxNy00YjVhLThiY2EtYzQ0YTdkOTNhMzk3In0=")
        neptune.create_experiment('optuna_test')
        neptune_callback = opt_utils.NeptuneCallback()
        study = optuna.create_study(direction='minimize')
        study.optimize(objective_AGLRs_second_step, n_trials=500, callbacks=[neptune_callback], n_jobs=1)
        print(study.best_params)
        print(study.best_value)
        print(study.best_trial)

    mat_data = sio.loadmat(DATABASE_NAME)
    emg_data = mat_data[DATABASE_TABLE]
    torque_data = emg_data[:, 6]
    emg_single_data = emg_data[:, DATA_COLUMN]

    result = emg_data[DATA_COLUMN, 7]
    print("Should be {0}".format(result))
    print("ONSET KOMI {0}".format(onset_komi(emg_single_data, 0.03)))
    print("ONSET TKVar {0}".format(onset_TKVar(emg_single_data, 200, 20)))
    print("ONSET BONATO {0}".format(onset_bonato(emg_single_data, 200, 7.74, 10, 25, 50)))
    print("ONSET SOLNIK {0}".format(onset_solnik(emg_single_data, 100, 0.03, 10)))
    print("ONSET SILVA {0}".format(onset_silva(emg_single_data, 40, 80, 0.02)))
    print("ONSET LONDRAL {0}".format(onset_londral(emg_single_data, 200, 0.05, 80)))
    print("ONSET HIDDEN FACTOR {0}".format(onset_hidden_factor(emg_single_data, 250, 100, 0.15)))

    # prepare_results(mat_data, [x for x in range(1, 30) if x not in [3, 4, 8, 11, 14, 19, 25]], 'after_change.csv')
    # create_statistics('after_change.csv')
    # onset_sign_changes(emg_single_data, 120, 1, 0.00724023569, True, "emg29-3combined")


def create_statistics(filename):
    def nan_to_none(x):
        if math.isnan(x):
            return None
        return x

    data = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        header = next(reader)
        for row in reader:
            data.append(row)
    parsed_data = zip(*data)
    list_1, list_2, list_3, list_4, list_5 = parsed_data
    results = [float(item) if item != '' else float('NaN') for item in list_2]
    sign_changes = [float(item) if item != '' else float('NaN') for item in list_3]
    AGLR_step = [float(item) if item != '' else float('NaN') for item in list_4]
    two_step = [float(item) if item != '' else float('NaN') for item in list_5]

    outlier = 500
    errors_sign_changes = [(sign_changes[i] - results[i]) if not math.isnan(sign_changes[i]) else float('NaN') for i in
                           range(0, len(results))]
    errors_AGLR_step = [(AGLR_step[i] - results[i]) if not math.isnan(AGLR_step[i]) else float('NaN') for i in
                        range(0, len(results))]
    errors_two_step = [(two_step[i] - results[i]) if not math.isnan(two_step[i]) else float('NaN') for i in
                       range(0, len(results))]
    test_list_aglr = [errors_AGLR_step[i] if errors_AGLR_step[i] != errors_two_step[i] else None for i in
                      range(0, len(errors_AGLR_step))]
    test_list_aglr = list(filter(None, test_list_aglr))
    test_list_two_step = [errors_two_step[i] if errors_AGLR_step[i] != errors_two_step[i] else None for i in
                          range(0, len(errors_AGLR_step))]
    test_list_two_step = list(filter(None, test_list_two_step))
    print(test_list_aglr)
    print(test_list_two_step)
    print("GROUPS - AGLR_step, two_step diffs")
    print(stats.kruskal(test_list_aglr, test_list_two_step, nan_policy="omit"))
    print("GROUPS - sign_changes, AGLR_step, two_step")
    print(stats.kruskal(errors_sign_changes, errors_AGLR_step, errors_two_step, nan_policy="omit"))
    print("GROUPS - sign_changes, AGLR_step")
    print(stats.kruskal(errors_sign_changes, errors_AGLR_step, nan_policy="omit"))
    print("GROUPS - AGLR_step, two_step")
    print(stats.kruskal(errors_AGLR_step, errors_two_step, nan_policy="omit"))
    print("GROUPS - sign_changes, two_step")
    print(stats.kruskal(errors_sign_changes, errors_two_step, nan_policy="omit"))

    print("SIGN CHANGES:")
    errors_sign_changes = list(map(nan_to_none, errors_sign_changes))
    print("No onset found - {0}".format(errors_sign_changes.count(None)))
    errors_sign_changes = list(filter(None, errors_sign_changes))
    print("ABSOLUTE ERROR MEAN: ")
    print(np.mean(np.abs(errors_sign_changes)))
    print("ERROR MEAN: ")
    print(np.mean(errors_sign_changes))
    print("VARIANCE: ")
    print(np.var(errors_sign_changes))
    make_histogram(errors_sign_changes, "sign_changes", False)
    print()

    print("AGLR STEP:")
    errors_AGLR_step = list(map(nan_to_none, errors_AGLR_step))
    print("No onset found - {0}".format(errors_AGLR_step.count(None)))
    errors_AGLR_step = list(filter(None, errors_AGLR_step))
    print("ABSOLUTE ERROR MEAN: ")
    print(np.mean(np.abs(errors_AGLR_step)))
    print("ERROR MEAN: ")
    print(np.mean(errors_AGLR_step))
    print("VARIANCE: ")
    print(np.var(errors_AGLR_step))
    make_histogram(errors_AGLR_step, "AGLR_step", False)
    print()

    print("TWO STEP:")
    errors_two_step = list(map(nan_to_none, errors_two_step))
    print("No onset found - {0}".format(errors_two_step.count(None)))
    errors_two_step = list(filter(None, errors_two_step))
    print("ABSOLUTE ERROR MEAN: ")
    print(np.mean(np.abs(errors_two_step)))
    print("ERROR MEAN: ")
    print(np.mean(errors_two_step))
    print("VARIANCE: ")
    print(np.var(errors_two_step))
    make_histogram(errors_two_step, "two_step", False)
    print()

    print("SIGN CHANGES (outlier = {0}):".format(outlier))
    outliers_size_sign_changes = len(errors_sign_changes)
    errors_sign_changes = [item for item in errors_sign_changes if abs(item) <= outlier]
    outliers_size_sign_changes -= len(errors_sign_changes)
    print(outliers_size_sign_changes)
    print("ABSOLUTE ERROR MEAN: ")
    print(np.mean(np.abs(errors_sign_changes)))
    print("ERROR MEAN: ")
    print(np.mean(errors_sign_changes))
    print("VARIANCE: ")
    print(np.var(errors_sign_changes))
    make_histogram(errors_sign_changes, "sign_changes_outlier", True)
    print()

    print("AGLR STEP (outlier = {0}):".format(outlier))
    outliers_size_aglr_step = len(errors_AGLR_step)
    errors_AGLR_step = [item for item in errors_AGLR_step if abs(item) <= outlier]
    outliers_size_aglr_step -= len(errors_AGLR_step)
    print(outliers_size_aglr_step)
    print("ABSOLUTE ERROR MEAN: ")
    print(np.mean(np.abs(errors_AGLR_step)))
    print("ERROR MEAN: ")
    print(np.mean(errors_AGLR_step))
    print("VARIANCE: ")
    print(np.var(errors_AGLR_step))
    make_histogram(errors_AGLR_step, "AGLR_step_outlier", True)
    print()

    print("TWO STEP (outlier = {0}):".format(outlier))
    outliers_size_two_step = len(errors_two_step)
    errors_two_step = [item for item in errors_two_step if abs(item) <= outlier]
    outliers_size_two_step -= len(errors_two_step)
    print(outliers_size_two_step)
    print("ABSOLUTE ERROR MEAN: ")
    print(np.mean(np.abs(errors_two_step)))
    print("ERROR MEAN: ")
    print(np.mean(errors_two_step))
    print("VARIANCE: ")
    print(np.var(errors_two_step))
    make_histogram(errors_two_step, "two_step_outlier", True)


def prepare_results(database, data_range, filename):
    def prepare_single_result(database, data_range, sign_changes, AGLR_step, two_step, make_plots=True):
        filenames = []
        results = []
        onsets_sign_changes = []
        onsets_AGLR_step = []
        onsets_two_step = []
        sign_changes_no_result_count = 0
        AGLR_step_no_result_count = 0
        two_step_no_result_count = 0
        for j in data_range:
            emg_data = database["emg{0}".format(j)]
            print("@@@@@@@@@@@@@@@@ {0}".format(j))
            for i in range(0, 6):
                emg_single_data = emg_data[:, i]
                result = emg_data[i, 7]
                if result > 0:
                    filename = "emg{0}-{1}".format(j, i)
                    try:
                        onset_sign_changes = \
                        sign_changes[0](emg_single_data, *sign_changes[1], print_plot=False, filename=filename)[0]
                    except:
                        onset_sign_changes = None
                    if onset_sign_changes is None:
                        sign_changes_no_result_count += 1
                    onsets_sign_changes.append(onset_sign_changes)

                    try:
                        print("AGLR STEP TIME")
                        time_before = time.time()
                        onset_AGLR_step = AGLR_step[0](emg_single_data, *AGLR_step[1])
                        print(time.time() - time_before)
                    except:
                        onset_AGLR_step = None
                    if onset_AGLR_step is None:
                        AGLR_step_no_result_count += 1
                    onsets_AGLR_step.append(onset_AGLR_step)

                    try:
                        print("TWO STEP TIME")
                        time_before = time.time()
                        onset_two_step = two_step[0](emg_single_data, *two_step[1])
                        print(time.time() - time_before)
                    except:
                        onset_two_step = None
                    if onset_two_step is None:
                        two_step_no_result_count += 1
                    onsets_two_step.append(onset_two_step)

                    # print("SHOULD BE {0}".format(result))
                    # print(onset_sign_changes)
                    # print(onset_AGLR_step)
                    # print(onset_two_step)
                    results.append(result)

                    filenames.append(filename)
                    if make_plots:
                        make_plot(emg_single_data, emg_data[:, 6], filename, result,
                                  onset_sign_changes,
                                  onset_AGLR_step,
                                  onset_two_step)

        sign_changes_errors = [(results[i] - onsets_sign_changes[i]) if onsets_sign_changes[i] is not None else None for
                               i in range(0, len(results))]
        sign_changes_result = {
            "function_name": sign_changes[0].__name__,
            "results": results,
            "found_onsets": onsets_sign_changes,
            "errors": sign_changes_errors,
            "errors_abs_mean": np.mean(np.abs(list(filter(None, sign_changes_errors)))),
            "no_result_count": sign_changes_no_result_count
        }

        AGLR_step_errors = [(results[i] - onsets_AGLR_step[i]) if onsets_AGLR_step[i] is not None else None for i in
                            range(0, len(results))]
        AGLR_step_result = {
            "function_name": AGLR_step[0].__name__,
            "results": results,
            "found_onsets": onsets_AGLR_step,
            "errors": AGLR_step_errors,
            "errors_abs_mean": np.mean(np.abs(list(filter(None, AGLR_step_errors)))),
            "no_result_count": AGLR_step_no_result_count
        }

        two_step_errors = [(results[i] - onsets_two_step[i]) if onsets_two_step[i] != None else None for i in
                           range(0, len(results))]
        two_step_result = {
            "function_name": two_step[0].__name__,
            "results": results,
            "found_onsets": onsets_two_step,
            "errors": two_step_errors,
            "errors_abs_mean": np.mean(np.abs(list(filter(None, two_step_errors)))),
            "no_result_count": two_step_no_result_count
        }
        return (filenames, results, sign_changes_result, AGLR_step_result, two_step_result)

    # {'h': 288, 'W': 69, 'M': 210}
    results = prepare_single_result(database, data_range, (onset_sign_changes, (120, 1, 0.00724023569)),
                                    (onset_AGLRstep, (241, 298, 239)),
                                    (onset_two_step_alg, (120, 1, 0.00724023569, 254, 10, 198)), make_plots=True)

    with open(filename, 'w', newline="") as result_file:
        wr = csv.writer(result_file, dialect='excel', delimiter=';')
        fieldnames = ["data_name", "real_onset", "sign_changes_onset", "AGLR_step_onset", "two_step_onset"]
        wr.writerow(fieldnames)
        list_1 = results[0]
        list_2 = results[1]
        list_3 = results[2]["found_onsets"]
        list_4 = results[3]["found_onsets"]
        list_5 = results[4]["found_onsets"]
        results = zip(list_1, list_2, list_3, list_4, list_5)
        for item in results:
            wr.writerow(item)


def split(list, n):
    k, m = divmod(len(list), n)
    return (list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def make_histogram(errors_data, filename, limited):
    def compute_histogram_bins(data, desired_bin_size, min_val=-500, max_val=500):
        min_boundary = -1.0 * (min_val % desired_bin_size - min_val)
        max_boundary = max_val - max_val % desired_bin_size + desired_bin_size
        n_bins = int((max_boundary - min_boundary) / desired_bin_size) + 1
        bins = np.linspace(min_boundary, max_boundary, n_bins)
        return bins

    fig, ax = plt.subplots()
    if limited:
        bins = compute_histogram_bins(errors_data, 50)
    else:
        bins = compute_histogram_bins(errors_data, 50, np.min(errors_data), np.max(errors_data))
    counts, bins, patches = plt.hist(errors_data, density=False, weights=np.ones(len(errors_data)) / len(errors_data),
                                     bins=bins, histtype='bar', edgecolor='black', linewidth=1.2)
    ax.set_xticks(bins)
    ax.set_title('{0} errors histogram'.format(filename))
    plt.xticks(bins, rotation='vertical', fontsize=12)
    plt.gcf().set_size_inches(19.2, 10.8)
    plt.ylabel('Probability')
    plt.xlabel('Error [ms]')
    fig.tight_layout()
    plt.savefig('./{0}_histogram.png'.format(filename))


def make_plot(emg_data, torque_data, filename, expected, found_onset_sign_changes=0, found_onset_AGLR_step=0,
              found_onset_two_step=0):
    """Creates a plot presenting voltage and torque"""
    fig, axs = plt.subplots(2)
    plt.style.use('seaborn-whitegrid')
    axs[0].plot(emg_data, linewidth=1)
    axs[0].set_xlim([0, len(emg_data)])
    axs[0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axs[0].set_title("EMG Signal", fontsize=18)
    axs[0].set_ylabel("EMG = [mV]", fontsize=16)
    axs[0].set_xlabel("t = [ms]", fontsize=16)
    axs[1].plot(torque_data, linewidth=1, color="red")
    axs[1].set_xlim([0, len(torque_data)])
    axs[1].set_ylim([0, max(torque_data) + 10])
    axs[1].set_title("Torque data", fontsize=18)
    axs[1].set_ylabel("Torque = [Nm]", fontsize=16)
    axs[1].set_xlabel("t = [ms]", fontsize=16)
    axs[0].axvline(x=expected, color='tab:green', alpha=0.5, linewidth=2,
                   label="real onset = {0}".format(round(expected, 2)))
    if found_onset_two_step is not None:
        axs[0].plot(found_onset_two_step, 0, 's', color='magenta', alpha=0.75, markersize=7,
                    label="two_step result = {0}".format(found_onset_two_step))
    if found_onset_AGLR_step is not None:
        axs[0].plot(found_onset_AGLR_step, 0, '^', color='yellow', alpha=0.85, markersize=8,
                    label="AGLR_step result = {0}".format(found_onset_AGLR_step))
    if found_onset_sign_changes is not None:
        axs[0].plot(found_onset_sign_changes, 0, 'ro', color='cyan', alpha=0.75, markersize=6,
                    label="sign_changes_result = {0}".format(found_onset_sign_changes))
    leg = axs[0].legend(loc='upper left', frameon=1)
    frame = leg.get_frame()
    frame.set_facecolor('lightgrey')
    frame.set_edgecolor('black')
    fig = plt.gcf()
    fig.set_size_inches(24, 12)
    plt.setp(axs, xticks=[i for i in range(0, len(emg_data) + 100, 100)])
    plt.setp(axs[0].get_xticklabels(), rotation=50, fontsize=14)
    plt.setp(axs[0].get_yticklabels(), fontsize=14)
    plt.setp(axs[1].get_xticklabels(), rotation=50, fontsize=14)
    plt.setp(axs[1].get_yticklabels(), fontsize=14)
    fig.tight_layout(pad=2)
    plt.savefig('./plots/{0}.svg'.format(filename), format='svg', dpi=300, bbox_inches='tight')


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
    best_param = (min(diffs_sum)[1])
    return best_param


if __name__ == "__main__":
    main()
