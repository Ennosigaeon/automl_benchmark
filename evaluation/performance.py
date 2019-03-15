from datetime import timedelta
from typing import List

import numpy as np

import benchmark
from adapter.base import BenchmarkResult
from benchmark import OpenML100Suite, OpenMLBenchmark
from evaluation.base import MongoPersistence
from evaluation.visualization import plot_incumbent_performance, plot_evaluated_configurations, \
    plot_evaluation_performance, plot_method_overhead


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


def print_openml_runtime():
    tasks = OpenML100Suite.tasks()

    ls = {
        'Random Search': [],
        'Grid Search': [],
        'SMAC': [],
        'BOHB': [],
        'Optunity': [],
        'hyperopt': [],
        'RoBo gp': [],
        'BTB': []
    }
    for i, id in enumerate(tasks):
        print('{}: {}'.format(i, id))
        benchmark = OpenMLBenchmark(id, load=False)

        d = {}
        results = persistence.load_all(benchmark)
        for res in results:
            for solver in res.solvers:
                d.setdefault(solver.algorithm, []).append(solver)

        for key, value in d.items():
            ls[key] += value

    for key, value in ls.items():
        v = np.array([solver.score for solver in value if solver.score < 1])

        evaluations = []
        for solver in value:
            values = [eval.score for eval in solver.evaluations]
            if min(values) < 1:
                evaluations += values
        evaluations = np.array(evaluations)

        runtime = np.array([solver.end - solver.start for solver in value if solver.score < 1]).mean()
        if key in ['Random Search', 'Grid Search', 'SMAC', 'BOHB', 'Optunity']:
            runtime *= 8
        else:
            runtime *= 4

        delta = timedelta(seconds=runtime.mean())

        print('{}: {:.4f} +- {:.4f}'.format(key, v.mean(), v.std()))
        print(str(delta))
        print('{}/{} = {:.4f}'.format(len(evaluations[evaluations == 1]), len(evaluations),
                                      len(evaluations[evaluations == 1]) / len(evaluations)))


if __name__ == '__main__':
    persistence = MongoPersistence('10.0.2.2', db='synthetic')
    ls = [benchmark.Levy(), benchmark.Branin(), benchmark.Hartmann6(), benchmark.Rosenbrock10D()]
    bm = benchmark.Levy()

    # noinspection PyUnreachableCode
    if True:
        for b in ls:
            res = persistence.load_all(b)
            print_best_incumbent(res)
            plot_incumbent_performance(res)
            plot_method_overhead(res)

    # noinspection PyUnreachableCode
    if False:
        res = persistence.load_single(bm)
        plot_evaluation_performance(res)

    # noinspection PyUnreachableCode
    if False:
        res = persistence.load_all(bm)
        plot_evaluated_configurations(res)

    # noinspection PyUnreachableCode
    if False:
        print_openml_runtime()
