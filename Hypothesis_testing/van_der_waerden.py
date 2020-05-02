import numpy as np
from scipy import stats


def van_der_waerden_statistic(x, y):
    x_extended = [(elem, 'x') for elem in x]
    y_extended = [(elem, 'y') for elem in y]
    w = np.concatenate((x_extended, y_extended))
    w = sorted(w, key=lambda elem: elem[0])

    statistic = 0
    variance = 0
    for i in range(len(w)):
        if w[i] == 'y':
            statistic += stats.norm.ppf(i + 1 / (len(x) + len(y) + 1))
        variance += stats.norm.ppf(i + 1 / (len(x) + len(y) + 1)) ** 2

    variance *= len(x) * len(y) / (len(x) + len(y)) / (len(x) + len(y) - 1)
    return statistic, variance


def van_der_waerden_test(x, y, alpha):
    quantile = stats.norm.ppf(1 - alpha / 2)
    statistic, variance = van_der_waerden_statistic(x, y)
    statistic /= variance
    return statistic < -quantile or statistic > quantile
