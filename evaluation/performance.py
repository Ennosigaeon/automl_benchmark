from typing import List

import numpy as np

import benchmark
from adapter.base import BenchmarkResult
from evaluation.base import MongoPersistence
from evaluation.visualization import plot_incumbent_performance, plot_evaluated_configurations, plot_evaluation_performance


def print_best_incumbent(ls: List[BenchmarkResult], iteration: int = -1):
    bm = ls[0].benchmark

    best = {}
    for solver in sum([b.solvers for b in ls], []):
        best.setdefault(solver.algorithm, []).append(
            abs(solver.incumbents[iteration].score - bm.get_meta_information()['f_opt'])
        )

    print(bm.get_meta_information()['name'])

    algorithms = ['Random Search', 'Grid Search', 'RoBo gp', 'hyperopt', 'SMAC', 'BOHB', 'BTB', 'Optunity']
    print(algorithms)
    for algorithm in algorithms:
        x = np.array(best[algorithm])
        print('& \\({:.2f} \\pm {:.2f}\\)\t'.format(x.mean(), x.std()), end='')
    print()


if __name__ == '__main__':
    persistence = MongoPersistence('10.0.2.2')

    # noinspection PyUnreachableCode
    if True:
        ls = [benchmark.Levy(), benchmark.Branin(), benchmark.Hartmann6(), benchmark.Rosenbrock10D()]
        for b in ls:
            res = persistence.load_all(b)
            print_best_incumbent(res)
            plot_incumbent_performance(res)

    # noinspection PyUnreachableCode
    if False:
        res = persistence.load_single(benchmark.Levy())
        plot_evaluation_performance(res)

    # noinspection PyUnreachableCode
    if False:
        res = persistence.load_all(benchmark.Branin())
        plot_evaluated_configurations(res)
