import numpy as np
import math
from numba import jit
from special import generate_array, normal_ppf


@jit(nopython=True)
def wilcoxon_statistic(x: np.ndarray, y: np.ndarray) -> float:
    """
    :param x: sample with non-shifted mean
    :param y: sample with shifted mean
    :return: Wilcoxon test statistic
    """
    w = generate_array(x, y)

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


@jit(nopython=True)
def wilcoxon_test(x: np.ndarray, y: np.ndarray, alpha: float) -> bool:
    """
    Performs Wilcoxon one-sided test
    :param x: sample with non-shifted mean
    :param y: sample with shifted mean
    :param alpha: significance level
    :return: True if H0 is rejected
    """
    statistic = wilcoxon_statistic(x, y)
    quantile = normal_ppf(1 - alpha)
    return statistic > quantile
