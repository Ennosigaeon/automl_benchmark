from typing import List

import matplotlib.pyplot as plt
import numpy as np

import benchmark
from adapter.base import BenchmarkResult
from evaluation.base import MongoPersistence


def plot_results(benchmark_result: BenchmarkResult):
    fig, ax = plt.subplots()
    fig.set_size_inches(16, 9)
    fig.set_dpi(250)

    f_opt = benchmark_result.benchmark.get_meta_information()['f_opt']
    ax.plot([0, 100], [f_opt, f_opt], 'k', label='Optimum')

    for res in benchmark_result.solvers:
        x, y = res.as_numpy(incumbent=False, x_axis='iterations')
        ax.plot(x, y, label=res.algorithm)

        x, y = res.as_numpy(incumbent=True, x_axis='iterations')
        ax.plot(x, y, label=res.algorithm)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    ax.set_title(benchmark_result.name)
    ax.set_xlabel('Iteration')

    plt.savefig('{}.pdf'.format(benchmark_result.name), bbox_inches="tight")
    fig.show()
    plt.show()


def plot_mulitple_results(ls: List[BenchmarkResult]):
    benchmark = ls[0].benchmark

    fig, ax = plt.subplots()
    fig.set_size_inches(16, 9)
    fig.set_dpi(250)
    # fig.tight_layout()

    f_opt = benchmark.get_meta_information()['f_opt']
    ax.plot([0, 100], [f_opt, f_opt], 'k', label='Optimum')

    solvers = {}
    for res in ls:
        for solver in res.solvers:
            if solver.algorithm not in solvers:
                solvers[solver.algorithm] = []
            x, y = solver.as_numpy(incumbent=True, x_axis='iterations')
            solvers[solver.algorithm].append(y)

    for name, values in solvers.items():
        y = np.vstack(values)
        mean = np.mean(y, axis=0)
        std = np.std(y, axis=0)

        ax.fill_between(np.arange(0, y.shape[1], 1), mean - std, mean + std, alpha=0.25)
        ax.plot(mean, label=name)

    ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.legend(loc='upper right')
    ax.set_title(ls[0].name)
    ax.set_xlabel('Iteration')

    plt.savefig('{}_aggregated.pdf'.format(ls[0].name), bbox_inches="tight")
    fig.show()
    plt.show()


if __name__ == '__main__':
    persistence = MongoPersistence('10.0.2.2')

    ls = [benchmark.Levy(), benchmark.Branin(), benchmark.Hartmann3(), benchmark.Hartmann6(), benchmark.Rosenbrock10D(),
          benchmark.Rosenbrock20D()]
    for b in ls:
        res = persistence.load_all(b)
        plot_mulitple_results(res)

    # res = persistence.load_single(benchmark.Levy())
    # plot_results(res)
