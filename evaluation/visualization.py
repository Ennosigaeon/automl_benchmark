from typing import List

import matplotlib.pyplot as plt
from hpolib.abstract_benchmark import AbstractBenchmark

import benchmark
from adapter.base import OptimizationStatistic
from evaluation.base import MongoPersistent


def plot_results(bench: AbstractBenchmark, results: List[OptimizationStatistic]):
    fig, ax = plt.subplots()

    ax.plot([0, 100], [bench.get_meta_information()['f_opt'], bench.get_meta_information()['f_opt']], 'k',
            label='Optimum')

    for res in results:
        # x, y = res.as_numpy(incumbent=False, x_axis='iterations')
        # ax.plot(x, y, label=res.algorithm)

        x, y = res.as_numpy(incumbent=True, x_axis='iterations')
        ax.plot(x, y, label=res.algorithm)

    # ax.set_xscale('log')
    ax.legend(loc='upper right')
    ax.set_title(bench.get_meta_information()['name'])

    plt.savefig('{}.png'.format(bench.get_meta_information()['name']))
    fig.show()
    plt.show()


if __name__ == '__main__':
    benchmark = benchmark.Levy()

    persistence = MongoPersistent('10.0.2.2')
    ls = persistence.load_results(benchmark)
    plot_results(benchmark, ls)
