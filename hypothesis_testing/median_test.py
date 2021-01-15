from special import generate_array
import numpy as np
from numba import jit
from special import normal_ppf


@jit(nopython=True)
def median_test_statistic(x: np.ndarray, y: np.ndarray) -> float:
    """
    :param x: sample with non-shifted mean
    :param y: sample with shifted mean
    :return: Median test statistic
    """
    w = generate_array(x, y)

    statistic = 0
    mid = (x.shape[0] + y.shape[0] + 1) / 2
    for i in range(len(w)):
        if w[i][1] == 1:
            statistic += 0.5 * (np.sign(i + 1 - mid) + 1)

    if (x.shape[0] + y.shape[0]) % 2 == 0:
        D = x.shape[0] * y.shape[0] / (4 * (x.shape[0] + y.shape[0] - 1))
    else:
        D = x.shape[0] * y.shape[0] / (4 * (x.shape[0] + y.shape[0]))
    return (statistic - y.shape[0] / 2) / np.sqrt(D)


@jit(nopython=True)
def median_test(x: np.ndarray, y: np.ndarray, alpha: float) -> bool:
    """
    Performs two sample one-sided median test
    :param x: sample with non-shifted mean
    :param y: sample with shifted mean
    :param alpha: significance level
    :return: True if H0 is rejected
    """
    quantile = normal_ppf(1 - alpha)
    statistic = median_test_statistic(x, y)
    return statistic > quantile
