from kolmogorov_test import kolmogorov_test
from student_test import student_test
from wilcoxon_test import wilcoxon_test
import sample_generator


if __name__ == "__main__":
    count = 0
    for i in range(10000):
        x = sample_generator.generate_normal(200, 0, 1)
        y = sample_generator.generate_normal(200, 0, 1)
        if wilcoxon_test(x, y, 0.05):
            count += 1
    print(count / 10000)
