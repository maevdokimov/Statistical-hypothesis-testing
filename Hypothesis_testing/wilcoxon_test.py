import numpy as np
import scipy.stats
import math
import sample_generator


def wilcoxon_statistic(x, y):
    x_extended = [(elem, 'x') for elem in x]
    y_extended = [(elem, 'y') for elem in y]
    w = np.concatenate((x_extended, y_extended))
    w = sorted(w, key=lambda elem: elem[0])

    result = 0
    for i in range(len(w)):
        if w[i][1] == 'y':
            result += i + 1
    mw = len(y) * (len(x) + len(y) + 1) / 2
    dw = len(x) * len(y) * (len(x) + len(y) + 1) / 12

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

    residual_sum = 0
    for elem in bounds:
        residual_sum += elem * (elem ** 2 - 1)
    dw -= len(x) * len(y) * residual_sum / (12 * (len(x) + len(y)) * (len(x) + len(y) - 1))
    if dw < 0:
        raise AssertionError("variance < 0 in wilcoxon test")

    return result, mw, dw


def wilcoxon_test(x, y, alpha):
    w, mw, dw = wilcoxon_statistic(x, y)
    standardized_w = (w - mw) / math.sqrt(dw)
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
