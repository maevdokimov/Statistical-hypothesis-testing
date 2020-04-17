import math
import numpy as np
import scipy.stats


def student_statistic(x, y):
    std = math.sqrt((len(x) * np.var(x) + len(y) * np.var(y)) / (len(x) + len(y) - 2))
    t_value = (np.mean(y) - np.mean(x)) / (std * math.sqrt(1 / len(x) + 1 / len(y)))
    return t_value


def student_test(x, y, alpha):
    quantile = scipy.stats.t.ppf(1 - alpha, len(x) + len(y) - 2)
    return student_statistic(x, y) > quantile
