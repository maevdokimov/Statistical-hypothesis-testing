import numpy as np
from numba import jit
import math

alpha = (10 - math.pi ** 2) / 5 / math.pi / (math.pi - 3)
beta = (120 - 60 * math.pi + 7 * math.pi * math.pi) / 15 / math.pi / (math.pi - 3)


@jit(nopython=True)
def generate_array(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Function unite two arrays to a sorted one with extra dimension containing 0 for 'x' and 1 for 'y'
    """
    x.sort()
    y.sort()
    w = np.zeros((x.shape[0] + y.shape[0], 2))
    x_pos = 0
    y_pos = 0
    for i in range(x.shape[0] + y.shape[0]):
        if x_pos == x.shape[0]:
            w[i][0] = y[y_pos]
            w[i][1] = 1
            y_pos += 1
        elif y_pos == y.shape[0]:
            w[i][0] = x[x_pos]
            x_pos += 1
        elif x[x_pos] < y[y_pos]:
            w[i][0] = x[x_pos]
            x_pos += 1
        else:
            w[i][0] = y[y_pos]
            w[i][1] = 1
            y_pos += 1

    return w


@jit(nopython=True)
def normal_ppf(p: float) -> float:
    """
    :param p: probability
    :return: PPF of standart normal random variable
    """
    Fa = p
    k = -math.log(1 - (2 * Fa - 1) ** 2)
    pi = math.pi
    D = (4 - pi * k * beta) ** 2 + 16 * pi * k * alpha
    quant = math.sqrt((pi * k * beta - 4 + math.sqrt(D)) / (4 * alpha))
    return quant if p > 0.5 else -quant
