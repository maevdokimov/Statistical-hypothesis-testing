import math
import numpy as np
from numba import jit


@jit(nopython=True)
def generate_normal(sample_size: int, mean: float, std: float) -> np.ndarray:
    return np.random.normal(mean, std, sample_size)


@jit(nopython=True)
def generate_t(sample_size: int, degs: int, mean: float, std: float) -> np.ndarray:
    return mean + std * math.sqrt((degs - 2) / degs) * np.random.standard_t(degs, sample_size)


@jit(nopython=True)
def generate_uniform(sample_size: int, mean: float, std: float) -> np.ndarray:
    interval_width = std * math.sqrt(12)
    low = mean - interval_width / 2
    high = mean + interval_width / 2
    return np.random.uniform(low, high, sample_size)


@jit(nopython=True)
def generate_tukey(sample_size: int, mean: float, std: float, std_scale: float) -> np.ndarray:
    result = np.zeros(sample_size)
    for i in range(sample_size):
        # define proc chance 5%
        if np.random.randint(20) == 0:
            result[i] = np.random.normal(mean, std * std_scale)
        else:
            result[i] = np.random.normal(mean, std)

    return result


@jit(nopython=True)
def generate_logistic(sample_size: int, mean: float, scale: float) -> np.ndarray:
    return np.random.logistic(mean, scale, sample_size)


@jit(nopython=True)
def generate_laplace(sample_size: int, mean: float, scale: float) -> np.ndarray:
    return np.random.laplace(mean, scale, sample_size)
