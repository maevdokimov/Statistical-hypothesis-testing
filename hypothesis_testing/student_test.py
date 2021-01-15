import math
import numpy as np
from special import normal_ppf
from numba import jit


@jit(nopython=True)
def student_statistic(x: np.ndarray, y: np.ndarray) -> float:
    """
    :param x: sample with non-shifted mean
    :param y: sample with shifted mean
    :return: Student t test statistic
    """
    std = math.sqrt((len(x) * np.var(x) + len(y) * np.var(y)) / (len(x) + len(y) - 2))
    statistic = (np.mean(y) - np.mean(x)) / (std * math.sqrt(1 / len(x) + 1 / len(y)))
    return statistic


@jit(nopython=True)
def student_test(x: np.ndarray, y: np.ndarray, alpha: float) -> bool:
    """
    Performs two sample one-sided Student t test
    :param x: sample with non-shifted mean
    :param y: sample with shifted mean
    :param alpha: significance level
    :return: True if H0 is rejected
    """
    # assuming each sample length >= 50
    quantile = normal_ppf(1 - alpha)
    return student_statistic(x, y) > quantile
