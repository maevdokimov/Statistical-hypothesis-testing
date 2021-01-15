import numpy as np
from numba import jit


@jit(nopython=True)
def kolmogorov_statistic_one_sided(x: np.ndarray, y: np.ndarray) -> float:
    """
    :param x: sample with non-shifted mean
    :param y: sample with shifted mean
    :return: p-value for one-sided Kolmogorov-Smirnov test
    """
    x.sort()
    y.sort()
    j = 0
    max_delta = -1
    for i in range(x.shape[0]):
        while j < y.shape[0] and y[j] < x[i]:
            j += 1

        if (i + 1 - j) / x.shape[0] > max_delta:
            max_delta = (i + 1 - j) / x.shape[0]

    m, n = max(x.shape[0], y.shape[0]), min(x.shape[0], y.shape[0])
    z = np.sqrt(m * n / (m + n)) * max_delta
    p = np.exp(-2 * z ** 2 - 2 * z * (m + 2 * n) / np.sqrt(m * n * (m + n)) / 3.0)
    if p < 0:
        return 0
    if p > 1:
        return 1

    return p


@jit(nopython=True)
def kolmogorov_test(x: np.ndarray, y:np.ndarray, alpha: float) -> bool:
    """
    Performs Kolmogorov-Smirnov two sample one-sided test
    :param x: sample with non-shifted mean
    :param y: sample with shifted mean
    :param alpha: significance level
    :return: True if H0 is rejected
    """
    p_value = kolmogorov_statistic_one_sided(x, y)
    return p_value < alpha

