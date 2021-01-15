from kolmogorov_test import kolmogorov_test
from median_test import median_test
from student_test import student_test
from wilcoxon_test import wilcoxon_test
from van_der_waerden import van_der_waerden_test
import sample_generator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from numba import jit

DELTA = 0.01
REPETITION = 10000
ALPHA = 0.05
FREEDOM_DEGREES = 5


@jit(nopython=True)
def do_test(distribution: str, test: str, sample_size: int) -> np.ndarray:
    """
    Do test with given distribution and sample size
    """
    results = np.zeros((1, 2))

    shift = 0
    while True:
        count = 0
        for i in range(REPETITION):
            if distribution == 'normal':
                x = sample_generator.generate_normal(sample_size, 0, 1)
                y = sample_generator.generate_normal(sample_size, shift, 1)
            elif distribution == 't':
                x = sample_generator.generate_t(sample_size, FREEDOM_DEGREES, 0, 1)
                y = sample_generator.generate_t(sample_size, FREEDOM_DEGREES, shift, 1)
            elif distribution == 'uniform':
                x = sample_generator.generate_uniform(sample_size, 0, 1)
                y = sample_generator.generate_uniform(sample_size, shift, 1)
            elif distribution == 'logistic':
                x = sample_generator.generate_logistic(sample_size, 0, 1)
                y = sample_generator.generate_logistic(sample_size, shift, 1)
            elif distribution == 'laplace':
                x = sample_generator.generate_laplace(sample_size, 0, 1)
                y = sample_generator.generate_laplace(sample_size, shift, 1)
            elif distribution == 'tukey':
                x = sample_generator.generate_tukey(sample_size, 0, 1, 10)
                y = sample_generator.generate_tukey(sample_size, shift, 1, 10)

            if test == 't':
                if student_test(x, y, ALPHA):
                    count += 1
            elif test == 'wilcoxon':
                if wilcoxon_test(x, y, ALPHA):
                    count += 1
            elif test == 'kolmogorov':
                if kolmogorov_test(x, y, ALPHA):
                    count += 1
            elif test == 'median':
                if median_test(x, y, ALPHA):
                    count += 1
            elif test == 'waerden':
                if van_der_waerden_test(x, y, ALPHA):
                    count += 1

        if shift == 0:
            results[0][0] = count / REPETITION
            results[0][1] = shift
        else:
            app = np.zeros((1, 2))
            app[0][0] = count / REPETITION
            app[0][1] = shift
            results = np.concatenate((results, app))
        shift += DELTA
        if results[-1, 0] == 1:
            break

    return results


if __name__ == "__main__":
    cnt = 0
    distribution = {0: 't', 1: 'normal', 2: 'tukey', 3: 'uniform', 4: 'logistic', 5: 'laplace'}
    sample_size = {0: 50, 1: 100, 2: 300, 3: 500}
    total = len(distribution.keys()) * len(sample_size.keys())
    for dist in distribution.keys():
        for size in sample_size.keys():
            print("{} of {} ready".format(cnt, total))
            result_kolm = do_test(distribution[dist], 'kolmogorov', sample_size[size])
            result_t = do_test(distribution[dist], 't', sample_size[size])
            result_wilcoxon = do_test(distribution[dist], 'wilcoxon', sample_size[size])
            result_van = do_test(distribution[dist], 'waerden', sample_size[size])
            result_median = do_test(distribution[dist], 'median', sample_size[size])

            fig, ax = plt.subplots()

            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))

            plt.xlabel('\u0398')
            plt.ylabel('\u03B2')
            #
            ax.plot(result_kolm[:, 1], result_kolm[:, 0], label='kolmogorov test')
            ax.plot(result_t[:, 1], result_t[:, 0], label='t test')
            ax.plot(result_wilcoxon[:, 1], result_wilcoxon[:, 0], label='wilcoxon test')
            ax.plot(result_van[:, 1], result_van[:, 0], label='van der waerden test')
            ax.plot(result_median[:, 1], result_median[:, 0], label='median test')
            ax.legend()

            fig.set_figwidth(20)
            fig.set_figheight(16)

            path = os.path.join('results', '{}_{}.png'.format(distribution[dist], str(sample_size[size])))
            plt.savefig(path, bbox_inches='tight')
            plt.show()
            cnt += 1
