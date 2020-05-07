import numpy as np
from special import generate_array, normal_ppf
from numba import jit


@jit(nopython=True)
def van_der_waerden_statistic(x: np.ndarray, y: np.ndarray) -> float:
    """
    :param x: sample with non-shifted mean
    :param y: sample with shifted mean
    :return: Van der Waerden test statistic
    """
    w = generate_array(x, y)

    statistic = 0
    variance = 0
    for i in range(len(w)):
        if w[i][1] == 1:
            statistic += normal_ppf((i + 1) / (len(x) + len(y) + 1))
        variance += normal_ppf((i + 1) / (len(x) + len(y) + 1)) ** 2

    variance *= len(x) * len(y) / (len(x) + len(y)) / (len(x) + len(y) - 1)
    return statistic / np.sqrt(variance)


@jit(nopython=True)
def van_der_waerden_test(x: np.ndarray, y: np.ndarray, alpha: float) -> bool:
    """
    Performs Van der Waerden two sample one-sided test
    :param x: sample with non-shifted mean
    :param y: sample with shifted mean
    :param alpha: significance level
    :return: True if H0 is rejected
    """
    quantile = normal_ppf(1 - alpha)
    statistic = van_der_waerden_statistic(x, y)
    return statistic > quantile
