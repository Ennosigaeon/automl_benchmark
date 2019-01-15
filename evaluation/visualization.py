from typing import List

import matplotlib.pyplot as plt

import benchmark
from adapter.base import OptimizationStatistic
from evaluation.base import MongoPersistent


def plot_results(results: List[OptimizationStatistic]):
    fig, ax = plt.subplots()

    for res in results:
        x, y = res.as_numpy(incumbent=True)
        ax.plot(x, y, label=res.algorithm)

    ax.set_xscale('log')
    ax.legend()
    plt.savefig("image.pdf")
    fig.show()
    plt.show()


if __name__ == "__main__":
    benchmark = benchmark.Hartmann3()

    persistence = MongoPersistent('10.0.2.2')
    ls = persistence.load_results(benchmark)
    plot_results(ls)
