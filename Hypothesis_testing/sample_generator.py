import math
import numpy as np


def generate_normal(sample_size, mean, std):
    return np.random.normal(mean, std, sample_size)


def generate_t(sample_size, degs, mean, std):
    return mean + std * math.sqrt((degs - 2) / degs) * np.random.standard_t(degs, sample_size)


def generate_uniform(sample_size, mean, std):
    interval_width = std * math.sqrt(12)
    low = mean - interval_width / 2
    high = mean + interval_width / 2
    return np.random.uniform(low, high, sample_size)


def generate_tukey(sample_size, mean, std, std_scale):
    result = np.zeros(sample_size)
    for i in range(sample_size):
        # define proc chance 5%
        if np.random.randint(20) == 0:
            result[i] = np.random.normal(mean, std * std_scale)
        else:
            result[i] = np.random.normal(mean, std)

    return result
