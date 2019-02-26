import logging
import time
import warnings
from argparse import Namespace

from hpolib.abstract_benchmark import AbstractBenchmark

import benchmark
from adapter.base import BenchmarkResult
from evaluation.base import MongoPersistence

logging.basicConfig(level=40)  # 10: debug; 20: info

warnings.simplefilter(action='ignore', category=FutureWarning)


def run(persistence: MongoPersistence, b: AbstractBenchmark):
    # db.Branin.drop()
    # db.Branin.find({}, {'solvers.incumbents': 0}).pretty()
    # db.Branin.count()

    config_dict = {
        'n_jobs': 4,
        'timeout': None,
        'iterations': 100,
        'seed': int(time.time()),

        'random_search': True,
        'grid_search': True,
        'smac': True,
        'hyperopt': True,
        'bohb': True,
        'robo': True,
        'optunity': True,
        'btb': True
    }
    config = Namespace(**config_dict)

    benchmark_result = BenchmarkResult(b, config.n_jobs, config.seed)
    persistence.store_new_run(benchmark_result)

    objective_time = None

    # Random Search
    if config.random_search:
        print('Start random search')
        from adapter.random_search import ObjectiveRandomSearch
        rs = ObjectiveRandomSearch(config.n_jobs, config.timeout, config.iterations, config.seed)
        stats = rs.optimize(b)
        benchmark_result.add_result(stats)
        persistence.store_results(benchmark_result, stats)

        # Estimate of objective time. Used to select iterations for fixed iterations procedures
        objective_time = stats.runtime['objective_function'][0]

        print('Finished after {}s'.format(stats.end - stats.start))
        print(stats)

    # Grid Search
    if config.grid_search:
        print('Start grid search')
        from adapter.grid_search import ObjectiveGridSearch
        gs = ObjectiveGridSearch(config.n_jobs, config.timeout, config.iterations)
        n = gs.estimate_grid_size(len(b.get_meta_information().get('bounds', [])), objective_time)
        print('Using grid size of {}'.format(n))
        stats = gs.optimize(b, n)
        benchmark_result.add_result(stats)
        persistence.store_results(benchmark_result, stats)
        print('Finished after {}s'.format(stats.end - stats.start))
        print(stats)

    # SMAC
    if config.smac:
        print('Start SMAC')
        from adapter.smac import SmacAdapter
        smac = SmacAdapter(config.n_jobs, config.timeout, config.iterations, config.seed)
        stats = smac.optimize(b, objective_time)
        benchmark_result.add_result(stats)
        persistence.store_results(benchmark_result, stats)
        print('Finished after {}s'.format(stats.end - stats.start))
        print(stats)

    # hyperopt
    if config.hyperopt:
        print('Start hyperopt')
        from adapter.hyperopt import HyperoptAdapter
        hyperopt = HyperoptAdapter(config.n_jobs, config.timeout, config.iterations, config.seed)
        stats = hyperopt.optimize(b)
        benchmark_result.add_result(stats)
        persistence.store_results(benchmark_result, stats)
        print('Finished after {}s'.format(stats.end - stats.start))
        print(stats)

    # bohb
    if config.bohb:
        print('Start bohb')
        from adapter.bohb import BohbAdapter
        bohb = BohbAdapter(config.n_jobs, config.timeout, config.iterations, config.seed)
        stats = bohb.optimize(b)
        benchmark_result.add_result(stats)
        persistence.store_results(benchmark_result, stats)
        print('Finished after {}s'.format(stats.end - stats.start))
        print(stats)

    # RoBo
    if config.robo:
        print('Start robo')
        from adapter.robo import RoBoAdapter
        robo = RoBoAdapter(config.n_jobs, config.timeout, config.iterations, config.seed)
        stats = robo.optimize(b, model_type='gp')
        benchmark_result.add_result(stats)
        persistence.store_results(benchmark_result, stats)
        print('Finished after {}s'.format(stats.end - stats.start))
        print(stats)

    # Optunity
    if config.optunity:
        print('Start optunity')
        from adapter.optunity_adapter import OptunityAdapter
        optunity = OptunityAdapter(config.n_jobs, config.timeout, config.iterations, config.seed)
        stats = optunity.optimize(b)
        benchmark_result.add_result(stats)
        persistence.store_results(benchmark_result, stats)
        print('Finished after {}s'.format(stats.end - stats.start))
        print(stats)

    # BTB
    if config.btb:
        print('Start btb')
        from adapter.btb_adapter import BtbAdapter
        btb = BtbAdapter(config.n_jobs, config.timeout, config.iterations, config.seed)
        stats = btb.optimize(b)
        benchmark_result.add_result(stats)
        persistence.store_results(benchmark_result, stats)
        print('Finished after {}s'.format(stats.end - stats.start))
        print(stats)


if __name__ == '__main__':
    persistence = MongoPersistence('10.0.2.2')
    b = benchmark.Levy()
    for i in range(10):
        run(persistence, b)