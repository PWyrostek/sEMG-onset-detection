import numpy as np
from utilities import make_plot_sign_changes


def onset_sign_changes(data, W, k=1.3, d=0.00724023569, print_plot=False, filename=""):
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

    signs = [1 if single_data >= 0 else -1 for single_data in data]
    mul = 1.5
    h = (max(np.abs(diff(data[0:int(W * mul)]))) + d) * k
    data_before_change = data
    data = abs(data)
    variability = []
    for i in range(W // 2, len(data) - W // 2):
        points = 0
        for j in range(i - W // 2 + 1, i + W // 2 - 1):
            if signs[j] == signs[j + 1] and (data[j] - data[j + 1]) > h:
                points += 1
        variability.append(points)
    if print_plot:
        variability_plot_data=([0] * (W//2)) + variability + ([0] * (W//2))
        make_plot_sign_changes(data_before_change, variability_plot_data, filename)
    if max(variability) == 0:
        return (None, None)
    return (find_left_side(), find_right_side())
