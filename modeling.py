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
import math
from scipy import stats
from onset_aglr import *
from onset_bonato import onset_bonato
from onset_hidden_factor import onset_hidden_factor
from onset_hodges_bui import onset_hodges_bui
from onset_komi import *
from onset_londral import onset_londral
from onset_silva import onset_silva
from onset_solnik import onset_solnik
from onset_tkvar import onset_TKVar
from onset_two_step import *
from credentials import project_name, personal_token
import statistics

pp = pprint.PrettyPrinter(indent=4)
THREADS_AMOUNT = 5

DATABASE_NAME = 'database.mat'
DATABASE_TABLE = 'emg1'
DATA_COLUMN = 0
OPTIMIZATION_TRIALS = 100
OPTIMIZATION_CONCURRENT_JOBS = 6

class ArgumentSearch:
    def __init__(self, name, min, max, is_int=True):
        self.name = name
        self.min = min
        self.max = max
        self.is_int = is_int


OPTIMIZATION_DATA = {
    onset_komi: [ArgumentSearch("h", 0.01, 0.3, False)],
    onset_TKVar: [ArgumentSearch("W", 40, 250, True), ArgumentSearch("g", 1, 50, True)],
    onset_hodges_bui: [ArgumentSearch("W", 40, 250, True), ArgumentSearch("h", 0.01, 10, False)],
    onset_solnik: [ArgumentSearch("W", 40, 250, True), ArgumentSearch("h", 0.001, 4, False), ArgumentSearch("duration", 10, 50, True)],
    onset_silva: [ArgumentSearch("W_1", 10, 200, True), ArgumentSearch("W_2", 10, 100, True), ArgumentSearch("h", 0.01, 0.5, False)],
    onset_londral: [ArgumentSearch("W", 40, 250, True), ArgumentSearch("h", 0.01, 10, False), ArgumentSearch("duration", 80, 120, True)],
    onset_hidden_factor: [ArgumentSearch("W", 40, 200, True), ArgumentSearch("h", 0.001, 1, False)],
    onset_bonato: [ArgumentSearch("h", 1, 20, False), ArgumentSearch("duration", 80, 120, True), ArgumentSearch("num_of_all_active", 20, 150, True), ArgumentSearch("pool_size", 20, 150, True)],
    onset_sign_changes: [ArgumentSearch("W", 50, 400, True), ArgumentSearch("k", 1, 3, True), ArgumentSearch("d", 0.0025, 0.03, False)],
    onset_AGLRstep: [ArgumentSearch("h", 1, 300, True), ArgumentSearch("W", 20, 250, True), ArgumentSearch("M", 50, 250, True)],
    "onset_two_step_first_step": [ArgumentSearch("W", 100, 400, True), ArgumentSearch("k", 1, 3, True), ArgumentSearch("d", 0.0025, 0.03, False)],
    "onset_two_step_second_step": ([ArgumentSearch("h", 1, 300, True), ArgumentSearch("W", 10, 100, True), ArgumentSearch("M", 50, 250, True)])
}


def main():
    def prepare_data():
        training_database_ids = [3, 4, 8, 11, 14, 19, 25]
        training_column_ids = [0, 5]
        training_data = []
        test_data = []
        for i in range(1, 30):
            for j in range(0, 6):
                data = mat_data['emg{0}'.format(i)][:, j]
                result = mat_data['emg{0}'.format(i)][j, 7]
                torque = mat_data['emg{0}'.format(i)][:, 6]
                identifier = 'emg{0}-{1}'.format(i, j)
                if i in training_database_ids and j in training_column_ids:
                    training_data.append((data, result, torque, identifier))
                else:
                    test_data.append((data, result, torque, identifier))
        return (training_data, test_data)


    def find_minimizing_params(function, arguments, first_step_args=()):
        def objective_function(trial):
            mapped_arguments = [trial.suggest_int(argument.name, argument.min,
                                                  argument.max) if argument.is_int else trial.suggest_uniform(
                argument.name, argument.min, argument.max) for argument in arguments]
            sum = 0
            for data in training_data:
                emg_single_data = data[0]
                try:
                    result = data[1]

                    if function == onset_sign_changes or function == "onset_two_step_first_step":
                        value, right_side = onset_sign_changes(emg_single_data, *mapped_arguments)
                    elif function == onset_two_step_alg:
                        value = function(emg_single_data, *first_step_args, *mapped_arguments)
                    else:
                        value = function(emg_single_data, *mapped_arguments)

                    sum += abs(value - result)
                    if function == "onset_two_step_first_step" and (value is None or value > result or right_side < result):
                        sum += 5000
                    if value == -1:
                        sum += 5000
                except:
                    sum += 5000
            cost = sum
            return cost

        if function == "onset_two_step_second_step":
            function = onset_two_step_alg
            arguments = arguments

        neptune.init(project_qualified_name=project_name, api_token=personal_token)
        neptune.create_experiment(name=function if isinstance(function, str) else function.__name__)
        neptune_callback = opt_utils.NeptuneCallback()
        study = optuna.create_study(direction='minimize')
        study.optimize(objective_function, n_trials=OPTIMIZATION_TRIALS, callbacks=[neptune_callback], n_jobs=OPTIMIZATION_CONCURRENT_JOBS)
        print(study.best_params)
        print(study.best_value)
        print(study.best_trial)
        return study.best_params

    mat_data = sio.loadmat(DATABASE_NAME)
    emg_data = mat_data[DATABASE_TABLE]
    torque_data = emg_data[:, 6]
    emg_single_data = emg_data[:, DATA_COLUMN]
    training_data, test_data = prepare_data()

    # minimzing_function = "onset_two_step_first_step"
    # find_minimizing_params(minimzing_function, OPTIMIZATION_DATA[minimzing_function])

    optimization_results = {}
    first_step_arguments = ()
    for key in OPTIMIZATION_DATA:
        key_name = key if isinstance(key, str) else key.__name__
        optimization_results[key_name] = find_minimizing_params(key, OPTIMIZATION_DATA[key], first_step_arguments if key_name == "onset_two_step_second_step" else ())
        if key_name == "onset_two_step_first_step":
            first_step_arguments = (optimization_results[key_name]['W'], optimization_results[key_name]['k'], optimization_results[key_name]['d'])

    print(optimization_results)

    # result = emg_data[DATA_COLUMN, 7]
    # print("Should be {0}".format(result))
    # print(onset_sign_changes(emg_single_data,225, 1, 0.00615049902779793))
    # print(onset_two_step_alg(emg_single_data, 399, 1, 0.013844505139074995, 253, 10, 59))
    # print("ONSET KOMI {0}".format(onset_komi(emg_single_data, 0.03)))
    # print("ONSET TKVar {0}".format(onset_TKVar(emg_single_data, 300, 50)))
    # print("ONSET BONATO {0}".format(onset_bonato(emg_single_data, 7.74, 10, 25, 50)))
    # print("ONSET SOLNIK {0}".format(onset_solnik(emg_single_data, 100, 0.03, 10)))
    # print("ONSET SILVA {0}".format(onset_silva(emg_single_data, 40, 80, 0.02)))
    # print("ONSET LONDRAL {0}".format(onset_londral(emg_single_data, 117, 0.07780019392534165, 120)))
    # print("ONSET HIDDEN FACTOR {0}".format(onset_hidden_factor(emg_single_data, 100, 0.15)))
    # print("ONSET HODGES BUI {0}".format(onset_hodges_bui(emg_single_data, 100, 3)))

    # prepare_results(test_data, 'after_change.csv')
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


def prepare_results(test_data, filename):
    def prepare_single_result(sign_changes, AGLR_step, two_step, make_plots=True):
        filenames = []
        results = []
        onsets_sign_changes = []
        onsets_AGLR_step = []
        onsets_two_step = []
        sign_changes_no_result_count = 0
        AGLR_step_no_result_count = 0
        two_step_no_result_count = 0
        for data in test_data:
            emg_single_data = data[0]
            result = data[1]
            if result > 0:
                filename = data[3]
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
                    make_plot(emg_single_data, data[2], filename, result,
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
    results = prepare_single_result((onset_sign_changes, (120, 1, 0.00724023569)),
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


if __name__ == "__main__":
    main()
