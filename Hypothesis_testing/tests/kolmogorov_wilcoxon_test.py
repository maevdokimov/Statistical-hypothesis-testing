from matplotlib import ticker
import os
from testing import do_test
import matplotlib.pyplot as plt

SAMPLE_SIZE = 300
DELTA = 0.01
REPETITION = 1000
ALPHA = 0.05
FREEDOM_DEGREES = 5

if __name__ == "__main__":
    result_kolm = do_test('normal', 'kolmogorov', SAMPLE_SIZE)
    result_wilcoxon = do_test('normal', 'wilcoxon', SAMPLE_SIZE)

    fig, ax = plt.subplots()

    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))

    plt.xlabel('\u0398')
    plt.ylabel('\u03B2')

    ax.plot(result_kolm[:, 1], result_kolm[:, 0], label='kolmogorov test')
    ax.plot(result_wilcoxon[:, 1], result_wilcoxon[:, 0], label='wilcoxon test')
    ax.legend()

    fig.set_figwidth(20)
    fig.set_figheight(16)

    path = os.path.join('images', 'kolmogorov_vs_wilcoxon', "graph_{}.png".format(SAMPLE_SIZE))
    plt.savefig(path, bbox_inches='tight')
    plt.show()
