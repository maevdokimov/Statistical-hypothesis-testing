from numba import jit
import numpy as np


def abc(x):
    return x


@jit
def test_numpy():
    return abc(1)


if __name__ == "__main__":
    test_numpy()
