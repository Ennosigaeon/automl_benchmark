from datetime import timedelta
from typing import List

import numpy as np

import benchmark
from adapter.base import BenchmarkResult
from benchmark import OpenML100Suite, OpenMLBenchmark
from evaluation.base import MongoPersistence
from evaluation.visualization import plot_incumbent_performance, plot_evaluated_configurations, \
    plot_evaluation_performance, plot_method_overhead, plot_openml_100, plot_branin, plot_successive_halving


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


def print_automl_framework_results():
    def print_res(ls):
        mean = []
        std = []
        for x in ls:
            a = np.array(x)
            mean.append(a.mean())
            std.append(2.5 * a.std())
            print('{:1.4f} \\(\\pm\\) {:1.4f}'.format(mean[-1], std[-1]))
        print('\n{:1.4f} \\(\\pm\\) {:1.4f}'.format(np.array(mean).mean(), np.array(std).mean()))

    tpot = [
        [0.033333333333333326, 0.01904761904761909, 0.023809523809523836, 0.042857142857142816],  # 15
        [0.4411764705882353, 0.42533936651583715, 0.4298642533936652, 0.43665158371040724],  # 23
        [0.38009049773755654, 0.35746606334841624, 0.33484162895927605, 0.35746606334841624],  # 2079
        [0.012367491166077715, 0.012367491166077715, 0.01855123674911663, 0.00883392226148405],  # 3021
        [0.8, 0.8041666666666667, 0.7875, 0.7875],  # 3560
        [0.34158415841584155, 0.28712871287128716, 0.3465346534653465, 0.32178217821782173],  # 3561
        [0.07379999999999998, 0.07220000000000004, 0.07066666666666666, 0.07306666666666661],  # 3946
        [0.3895833333333333, 0.39375000000000004, 0.41041666666666665, 0.45625000000000004],  # 9955
        [0.3730936819172114, 0.3943355119825708, 0.383442265795207, 0.3883442265795207],  # 9985
        [0.3153274814314653, 0.312288993923025, 0.337609723160027, 0.34604996623902773],  # 14969
        [],  # 29
        [],  # 41
        [],  # 3904
        [],  # 7592
        [],  # 146606
    ]

    hpsklearn = [
        [1, 1, 1],  # Missing values  # 15
        [0.45927601809954754, 0.4638009049773756, 0.47285067873303166],  # 23
        [1, 1, 1],  # Too large for float32  # 2079
        [1, 1, 1],  # Too large for float32  # 3021
        [0.825, 0.7833333333333333, 0.8166666666666667],  # 3560
        [1, 1, 1],  # Too large for float32  # 3561
        [1, 1, 1],  # Too large for float32  # 3946
        [0.6854166666666667, 0.58125],  # 9955
        [0.58125, 0.6854166666666667, 0.39324618736383443],  # 9985
        [0.47984749455337694, 0.6076975016880486, 0.30756245779878455],  # 14969
        [],  # 29
        [],  # 41
        [],  # 3904
        [],  # 7592
        [],  # 146606
    ]

    auto_sklearn = [
        [0.014285714285714235, 0.02857142857142858, 0.033333333333333326, 0.02857142857142858],  # 15
        [0.4411764705882353, 0.4683257918552036, 0.4321266968325792, 0.4638009049773756],  # 23
        [0.36651583710407243, 0.330316742081448, 0.38009049773755654, 0.3393665158371041],  # 2079
        [0.012367491166077715, 0.02296819787985871, 0.013250883392226132, 0.021201413427561877],  # 3021
        [0.7333333333333334, 0.8125, 0.8083333333333333, 0.7541666666666667],  # 3560
        [0.33168316831683164, 0.3564356435643564, 0.38613861386138615, 0.3168316831683168],  # 3561
        [0.07340000000000002, 0.07399999999999995, 0.07166666666666666, 0.07279999999999998],  # 3946
        [0.35, 0.37916666666666665, 0.35624999999999996, 0.35624999999999996, 0.38541666666666663],  # 9955
        [0.4128540305010894, 0.4046840958605664, 0.4041394335511983, 0.3899782135076253],  # 9985
        [0.3200540175557056, 0.33524645509790685, 0.34301147873058746, 0.32680621201890614],  # 14969
        [],  # 29
        [],  # 41
        [],  # 3904
        [],  # 7592
        [],  # 146606
    ]

    atm = [
        [0.0049261083743842304, 0.014563106796116498, 0.02857142857142858, 0.00952380952380949],  # 15
        [0.4683257918552036, 0.45927601809954754, 0.4570135746606335, 0.420814479638009],  # 23
        [0.3869346733668342, 0.41116751269035534, 0.3529411764705882, 0.3755656108597285],  # 2079
        [0.026501766784452263, 0.028268551236749095, 0.03180212014134276, 0.022084805653710293],  # 3021
        [0.75, 0.7375, 0.75, 0.7666666666666666],  # 3560
        [1, 1, 0.306930693069307, 0.25742574257425743],  # ???  # 3561
        [1, 1, 1, 1],  # Memory Error  # 3946
        [0.9791666666666666, 0.9770833333333333, 0.883333333333333, 0.84166666666666673],  # 9955
        [0.5784313725490196, 0.470315022081448, 0.5125272331154684, 0.8730936819172113],  # 9985
        [0.712356515867657, 0.700202565833896, 1, 1],  # 14969
        [],  # 29
        [],  # 41
        [],  # 3904
        [],  # 7592
        [],  # 146606
    ]
    print('auto_sklearn')
    print_res(auto_sklearn)
    print('TPOT')
    print_res(tpot)
    print('ATM')
    print_res(atm)
    print('hpsklearn')
    print_res(hpsklearn)


if __name__ == '__main__':
    persistence = MongoPersistence('10.0.2.2', db='synthetic')
    ls = [benchmark.Levy(), benchmark.Branin(), benchmark.Hartmann6(), benchmark.Rosenbrock10D()]
    bm = benchmark.Branin()

    plot_successive_halving()
    plot_branin()

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

    if True:
        plot_openml_100(persistence)
