import scipy.stats


def kolmogorov_test(x, y, alpha):
    p_value = scipy.stats.ks_2samp(x, y, alternative='greater')[1]
    return p_value < alpha
