import numpy as np
import scipy.stats
import math
import sample_generator
from numba import jit


@jit(nopython=True)
def wilcoxon_statistic(x, y):
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

    result = 0
    for i in range(len(w)):
        if w[i][1] == 1:
            result += i + 1
    mw = len(y) * (len(x) + len(y) + 1) / 2
    dw = len(x) * len(y) * (len(x) + len(y) + 1) / 12.

    prev_elem = w[0][0]
    bound_length = 1
    bounds = []
    for i in range(1, len(w)):
        if w[i][0] == prev_elem:
            bound_length += 1
        else:
            bounds.append(bound_length)
            prev_elem = w[i][0]
            bound_length = 1
    if bound_length > 1:
        bounds.append(bound_length)

    return (result - mw) / math.sqrt(dw)


# @jit(nopython=True)
def wilcoxon_test(x, y, alpha):
    # w, mw, dw = wilcoxon_statistic(x, y)
    # standardized_w = (w - mw) / math.sqrt(dw)
    standardized_w = wilcoxon_statistic(x, y)
    quantile = scipy.stats.norm.ppf(1 - alpha)
    return standardized_w > quantile


if __name__ == "__main__":
    # x = np.random.normal(0, 1, 100)
    # y = np.random.normal(500, 1, 100)
    x = sample_generator.generate_t(100, 5, 0, 1)
    y = sample_generator.generate_t(100, 5, 500, 1)
    w, mw, dw = wilcoxon_statistic(x, y)
    standardized_w = (w - mw) / math.sqrt(dw)
    print(standardized_w)
    print(scipy.stats.norm.ppf(1 - 0.01))
