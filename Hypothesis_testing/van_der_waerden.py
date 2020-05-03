import numpy as np
from scipy import stats
from numba import jit


@jit(nopython=True)
def generate_array(x, y):
    x.sort()
    y.sort()
    w = np.zeros((x.shape[0] + y.shape[0], 2))
    x_pos = 0
    y_pos = 0
    for i in range(x.shape[0] + y.shape[0]):
        if x_pos == x.shape[0]:
            w[i][0] = y[y_pos]
            w[i][1] = 1
            y_pos += 1
        elif y_pos == y.shape[0]:
            w[i][0] = x[x_pos]
            x_pos += 1
        elif x[x_pos] < y[y_pos]:
            w[i][0] = x[x_pos]
            x_pos += 1
        else:
            w[i][0] = y[y_pos]
            w[i][1] = 1
            y_pos += 1

    return w


# @jit(nopython=True)
def van_der_waerden_statistic(x, y):
    w = generate_array(x, y)

    statistic = 0
    variance = 0
    for i in range(len(w)):
        if w[i][1] == '1':
            statistic += stats.norm.ppf((i + 1) / (len(x) + len(y) + 1))
        variance += stats.norm.ppf((i + 1) / (len(x) + len(y) + 1)) ** 2

    variance *= len(x) * len(y) / (len(x) + len(y)) / (len(x) + len(y) - 1)
    return statistic, variance


# @jit(nopython=True)
def van_der_waerden_test(x, y, alpha):
    quantile = stats.norm.ppf(1 - alpha / 2)
    statistic, variance = van_der_waerden_statistic(x, y)
    statistic /= np.sqrt(variance)
    return statistic < -quantile or statistic > quantile
