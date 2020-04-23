from kolmogorov_test import kolmogorov_test
from student_test import student_test
from wilcoxon_test import wilcoxon_test
import sample_generator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

DELTA = 0.01
REPETITION = 100
ALPHA = 0.05
FREEDOM_DEGREES = 5


def do_test(distribution, test, sample_size):
    results = np.empty([1, 2])

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
            else:
                raise NotImplemented('No such distribution ' + distribution)

            if test == 't':
                if student_test(x, y, ALPHA):
                    count += 1
            elif test == 'wilcoxon':
                if wilcoxon_test(x, y, ALPHA):
                    count += 1
            elif test == 'kolmogorov':
                if kolmogorov_test(x, y, ALPHA):
                    count += 1
            else:
                raise NotImplemented('No such test ' + test)

        if shift == 0:
            results[0] = [count / REPETITION, shift]
        else:
            results = np.vstack((results, np.array([count / REPETITION, shift])))
        shift += DELTA
        if results[-1, 0] == 1:
            break

    return results


if __name__ == "__main__":
    distribution = {0: 't', 1: 'normal', 2: 'tukey', 3: 'uniform', 4: 'logistic', 5: 'laplace'}
    sample_size = {0: 50, 1: 100, 2: 300, 3: 500}
    for dist in range(6):
        for size in range(4):
            result_kolm = do_test(distribution[dist], 'kolmogorov', sample_size[size])
            result_t = do_test(distribution[dist], 't', sample_size[size])
            result_wilcoxon = do_test(distribution[dist], 'wilcoxon', sample_size[size])

            fig, ax = plt.subplots()

            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))

            plt.xlabel('\u0398')
            plt.ylabel('\u03B2')

            ax.plot(result_kolm[:, 1], result_kolm[:, 0], label='kolmogorov test')
            ax.plot(result_t[:, 1], result_t[:, 0], label='t test')
            ax.plot(result_wilcoxon[:, 1], result_wilcoxon[:, 0], label='wilcoxon test')
            ax.legend()

            fig.set_figwidth(20)
            fig.set_figheight(16)

            path = os.path.join('images', distribution[dist] + '_' + str(sample_size[size]) + '_' + '.png')
            plt.savefig(path, bbox_inches='tight')
            plt.show()
