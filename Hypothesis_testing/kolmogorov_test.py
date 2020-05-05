import numpy as np
import scipy.stats
import math
import sample_generator
from numba import jit


@jit(nopython=True)
def kolmogorov_statistic_one_sided(x, y):
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
    expt = -2 * z ** 2 - 2 * z * (m + 2 * n) / np.sqrt(m * n * (m + n)) / 3.0
    prob = np.exp(expt)
    if prob < 0:
        return 0
    if prob > 1:
        return 1
    return prob


def kolmogorov_test(x, y, alpha):
    # p_value = scipy.stats.ks_2samp(x, y, alternative='greater', mode='asymp')[1]
    # p_value = scipy.stats.ks_2samp(x, y, mode='asymp')[1]
    p_value = kolmogorov_statistic_one_sided(x, y)
    return p_value < alpha


if __name__ == "__main__":
    x = np.random.normal(0, 1, 100)
    y = np.random.normal(0, 1, 100)
    # x = sample_generator.generate_t(100, 5, 0, 1)
    # y = sample_generator.generate_t(100, 5, 500, 1)
    # w, mw, dw = wilcoxon_statistic(x, y)
    # standardized_w = (w - mw) / math.sqrt(dw)
    # print(standardized_w)
    # print(scipy.stats.norm.ppf(1 - 0.01))
    print(kolmogorov_statistic_one_sided(x, y))
    print(scipy.stats.ks_2samp(x, y, alternative='greater', mode='asymp')[1])
