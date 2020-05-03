import scipy.stats
from numba import jit


# @jit(nopython=True)
def kolmogorov_test(x, y, alpha):
    p_value = scipy.stats.ks_2samp(x, y, alternative='greater', mode='asymp')[1]
    # p_value = scipy.stats.ks_2samp(x, y, mode='asymp')[1]
    return p_value < alpha
