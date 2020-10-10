import csv
import multiprocessing
import os
import pprint
import time

import matplotlib.pyplot as plt
import neptune
import neptunecontrib.monitoring.optuna as opt_utils
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
import json
from datetime import datetime
import itertools

pp = pprint.PrettyPrinter(indent=4)
THREADS_AMOUNT = 5

DATABASE_NAME = 'database.mat'
OPTIMIZATION_TRIALS = 5
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
        full_data_dict = {}
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
                full_data_dict[identifier] = (data, result, torque)
        return (training_data, test_data, full_data_dict)

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
        study.optimize(objective_function, n_trials=OPTIMIZATION_TRIALS, callbacks=[neptune_callback],
                       n_jobs=OPTIMIZATION_CONCURRENT_JOBS)
        print(study.best_params)
        print(study.best_value)
        print(study.best_trial)
        return study.best_params

    mat_data = sio.loadmat(DATABASE_NAME)
    training_data, test_data, full_data_dict = prepare_data()

    ## for testing
    # database_table = 'emg27'
    # data_column = 4
    # emg_data = mat_data[database_table]
    # torque_data = emg_data[:, 6]
    # emg_single_data = emg_data[:, data_column]
    # print(onset_AGLRstep(emg_single_data, 241, 298, 239))

    # optimization_results = {}
    # first_step_arguments = ()
    # for key in OPTIMIZATION_DATA:
    #     key_name = key if isinstance(key, str) else key.__name__
    #     optimization_results[key_name] = find_minimizing_params(key, OPTIMIZATION_DATA[key],
    #                                                             first_step_arguments if key_name == "onset_two_step_second_step" else ())
    #     if key_name == "onset_two_step_first_step":
    #         first_step_arguments = (optimization_results[key_name]['W'], optimization_results[key_name]['k'],
    #                                 optimization_results[key_name]['d'])
    #
    # print(optimization_results)
    # current_date_time = datetime.now().strftime("%d-%m-%Y_%H%M%S")
    # with open("{0}_parameters.json".format(current_date_time), 'w') as outfile:
    #     json.dump(optimization_results, outfile)

    # prepare_results(test_data, "05-10-2020_20000.json", "refactor_test4.csv")
    make_plots('refactor_test3.csv', full_data_dict, True)
    # create_statistics('refactor_test3.csv')


def load_data(filename):
    data = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        header = next(reader)
        for row in reader:
            data.append(row)

    parsed_data = zip(*data)
    data_dict = dict(zip(header, parsed_data))
    return data_dict


def create_statistics(filename):
    def nan_to_none(x):
        if math.isnan(x):
            return None
        return x

    onsets_dict = load_data(filename)
    onsets_dict.pop('data_name')
    results = onsets_dict.pop('real_onset')
    results = [float(item) if item != '' else float('NaN') for item in results]
    data_length = len(results)
    errors_dict = {}
    outlier = 500
    for key in onsets_dict:
        onsets_dict[key] = [float(item) if item != '' else float('NaN') for item in onsets_dict[key]]
        errors_dict[key] = [(onsets_dict[key][i] - results[i]) if not math.isnan(onsets_dict[key][i]) else float('NaN')
                            for i in range(0, data_length)]

    kruskal_tests = []
    kruskal_titles = [None]
    for key in errors_dict:
        kruskal_titles.append(key)
    kruskal_tests.append(kruskal_titles)

    for key in errors_dict:
        kruskal_values = [key]
        for key_second in errors_dict:
            if key != key_second:
                kruskal_stats = stats.kruskal(errors_dict[key], errors_dict[key_second], nan_policy="omit")
                pvalue = kruskal_stats[1]
                kruskal_values.append(pvalue)
            else:
                kruskal_values.append(None)
        kruskal_tests.append(kruskal_values)

    filename_no_ext = os.path.splitext(filename)[0]
    kruskal_filename = filename_no_ext + " _kruskal_stats.csv"
    with open(kruskal_filename, 'w', newline="") as result_file:
        wr = csv.writer(result_file, dialect='excel', delimiter=';')
        for row in kruskal_tests:
            wr.writerow(row)

    statistics_filename = filename_no_ext + " _statistics.txt"
    with open(statistics_filename, 'w', newline="") as result_file:
        for key in errors_dict:
            print(key)
            print("{0} STATISTICS".format(key), file=result_file)
            errors = list(map(nan_to_none, errors_dict[key]))
            print("No onset found - {0}".format(errors.count(None)), file=result_file)
            errors = list(filter(None, errors))
            print("ABSOLUTE ERROR MEAN: ", file=result_file)
            print(np.mean(np.abs(errors)), file=result_file)
            print("ERROR MEAN: ", file=result_file)
            print(np.mean(errors), file=result_file)
            print("VARIANCE: ", file=result_file)
            print(np.var(errors), file=result_file)
            make_histogram(errors, key, False)
            print(file=result_file)
            print("{0} (outlier = {1}) STATISTICS".format(key, outlier), file=result_file)
            outliers_size = len(errors)
            errors = [item for item in errors if abs(item) <= outlier]
            outliers_size -= len(errors)
            print("RESULTS DROPPED: ", file=result_file)
            print(outliers_size, file=result_file)
            print("ABSOLUTE ERROR MEAN: ", file=result_file)
            print(np.mean(np.abs(errors)), file=result_file)
            print("ERROR MEAN: ", file=result_file)
            print(np.mean(errors), file=result_file)
            print("VARIANCE: ", file=result_file)
            print(np.var(errors), file=result_file)
            make_histogram(errors, "{0}_outlier".format(key), True)
            print(file=result_file)


def make_plots(filename, full_data_dict, show_torque=True):
    onsets_dict = load_data(filename)
    data_names = onsets_dict.pop('data_name')
    results = onsets_dict.pop('real_onset')
    for i in range(0, len(data_names)):
        data_name = data_names[i]
        print(data_name)
        emg_single_data = full_data_dict[data_name][0]
        torque_data = full_data_dict[data_name][2]
        result = float(results[i])
        values = {}
        for key in onsets_dict:
            value = onsets_dict[key][i]
            if value != '':
                values[key] = float(value)
        make_plot(emg_single_data, torque_data, data_name, result, values, show_torque)


def prepare_results(test_data, parameters_filename, filename):
    def load_params():
        params_dict = {}
        with open(parameters_filename) as json_file:
            data = json.load(json_file)
            for p in data:
                params_dict[p] = tuple(data[p].values())  # converts dictionary values to tuple
        return params_dict

    def get_function(name):
        if name == "onset_two_step_first_step":
            name = "onset_sign_changes"
        elif name == "onset_two_step_second_step":
            name = "onset_two_step_alg"
        return globals()[name]

    def get_value(key):
        function = get_function(key)
        parameters = params[key]
        try:
            if key == "onset_two_step_second_step":
                first_step_parameters = params["onset_two_step_first_step"]
                value = function(emg_single_data, *first_step_parameters, *parameters)
            elif function.__name__ == "onset_sign_changes":
                value = function(emg_single_data, *parameters)[0]
            else:
                value = function(emg_single_data, *parameters)
        except Exception as e:
            return None
        if value == -1 or value is None:
            return None
        return value

    params = load_params()
    with open(filename, 'w', newline="") as result_file:
        wr = csv.writer(result_file, dialect='excel', delimiter=';')
        fieldnames = ["data_name", "real_onset"] + list(params.keys())
        wr.writerow(fieldnames)

        for data in test_data:
            data_name = data[3]
            print("Current data: {0}".format(data_name))
            emg_single_data = data[0]
            result = data[1]
            if result > 0:
                values = {}
                for key in params:
                    values[key] = get_value(key)
            result_row = [data_name, result] + list(values.values())
            wr.writerow(result_row)


def split(list, n):
    k, m = divmod(len(list), n)
    return (list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def make_histogram(errors_data, filename, limited):
    def compute_histogram_bins(data, desired_bin_size, min_val=-500, max_val=500):
        min_boundary = -1.0 * (min_val % desired_bin_size - min_val)
        max_boundary = max_val - max_val % desired_bin_size + desired_bin_size
        n_bins = int((max_boundary - min_boundary) / desired_bin_size) + 1
        bins = np.linspace(min_boundary, max_boundary, n_bins)
        return bins[0:-1]

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
    if not os.path.exists('histograms'):
        os.makedirs('histograms')
    plt.savefig('./histograms/{0}_histogram.png'.format(filename))


def make_plot(emg_data, torque_data, filename, expected, onsets, show_torque=True):
    """Creates a plot presenting voltage and torque"""
    markers = itertools.cycle(('+', 'o', '*', 's', '^', 'v', "H", 'D', 'p', 'P', '<', '>'))
    colors = itertools.cycle(('tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
                              'tab:olive', 'tab:cyan', 'navy', 'turquoise', 'lightsalmon'))
    fig, axs = plt.subplots(2)
    plt.style.use('seaborn-whitegrid')
    axs[0].plot(emg_data, linewidth=1, clip_on=False, zorder=100)
    axs[0].set_xlim([0, len(emg_data)])
    axs[0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axs[0].set_title("EMG Signal", fontsize=18)
    axs[0].set_ylabel("EMG = [mV]", fontsize=16)
    axs[0].set_xlabel("t = [ms]", fontsize=16)
    axs[0].axvline(x=expected, color='tab:green', alpha=0.5, linewidth=2,
                   label="real onset = {0}".format(round(expected, 2)))
    for key in onsets:
        found_onset = onsets[key]
        if found_onset is not None:
            axs[0].plot(found_onset, 0, marker=next(markers), color=next(colors), alpha=0.85, markersize=9,
                        label="{0} result = {1}".format(key, found_onset), markeredgecolor='black', clip_on=False, zorder=101)

    if show_torque:
        axs[1].plot(torque_data, linewidth=1, color="red", clip_on=False, zorder=100)
        axs[1].set_xlim([0, len(torque_data)])
        axs[1].set_ylim([0, max(torque_data) + 10])
        axs[1].set_title("Torque data", fontsize=18)
        axs[1].set_ylabel("Torque = [Nm]", fontsize=16)
        axs[1].set_xlabel("t = [ms]", fontsize=16)
        plt.setp(axs[1].get_xticklabels(), rotation=50, fontsize=14)
        plt.setp(axs[1].get_yticklabels(), fontsize=14)
    else:
        fig.delaxes(axs[1])

    leg = axs[0].legend(loc='upper left', frameon=1, prop={'size': 6}, labelspacing=0.8, markerscale=0.8)
    frame = leg.get_frame()
    frame.set_facecolor('lightgrey')
    frame.set_edgecolor('black')
    fig = plt.gcf()
    fig.set_size_inches(24, 12)
    plt.setp(axs, xticks=[i for i in range(0, len(emg_data) + 100, 100)])
    plt.setp(axs[0].get_xticklabels(), rotation=50, fontsize=14)
    plt.setp(axs[0].get_yticklabels(), fontsize=14)

    fig.tight_layout(pad=2)
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig('./plots/{0}.svg'.format(filename), format='svg', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()
