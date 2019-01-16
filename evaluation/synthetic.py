import logging
from argparse import Namespace

import benchmark
from adapter.bohb import BohbAdapter
from adapter.grid_search import ObjectiveGridSearch
from adapter.hyperopt import HyperoptAdapter
from adapter.random_search import ObjectiveRandomSearch
from adapter.robo import RoBoAdapter
from adapter.smac import SmacAdapter
from evaluation.base import MongoPersistent
from evaluation.visualization import plot_results

logging.basicConfig(level=40)  # 10: debug; 20: info
persistence = MongoPersistent('10.0.2.2')

config_dict = {
    'n_jobs': 4,
    'timeout': 30,
    'seed': 42,

    'random_search': True,
    'grid_search': True,
    'smac': True,
    'hyperopt': True,
    'bohb': True,
    'robo': False
}
config = Namespace(**config_dict)

benchmark = benchmark.Hartmann3()
ls = []
# persistence.clear_old_results(benchmark)

objective_time = None

# Random Search
if config.random_search:
    print('Start random search')
    rs = ObjectiveRandomSearch(config.timeout, config.n_jobs)
    stats = rs.optimize(benchmark)
    persistence.store_results(benchmark, stats)

    # Estimate of objective time. Used to select iterations for fixed iterations procedures
    objective_time = stats.runtime['objective_function'][0]

    ls.append(stats)
    print('Finished after {}s'.format(stats.end - stats.start))
    print(stats)

# Grid Search
if config.grid_search:
    print('Start grid search')
    gs = ObjectiveGridSearch(config.timeout, config.n_jobs)
    n = gs.estimate_grid_size(objective_time, len(benchmark.get_meta_information()['bounds']))
    print('Using grid size of {}'.format(n))
    stats = gs.optimize(benchmark, n)
    persistence.store_results(benchmark, stats)
    ls.append(stats)
    print('Finished after {}s'.format(stats.end - stats.start))
    print(stats)

# SMAC
if config.smac:
    print('Start SMAC')
    smac = SmacAdapter(config.timeout, config.n_jobs, config.seed)
    stats = smac.optimize(benchmark)
    persistence.store_results(benchmark, stats)
    ls.append(stats)
    print('Finished after {}s'.format(stats.end - stats.start))
    print(stats)

# hyperopt
if config.hyperopt:
    print('Start hyperopt')
    hyperopt = HyperoptAdapter(config.timeout, config.n_jobs)
    iterations = hyperopt.estimate_iterations(objective_time)
    print('Using maximal {} iterations'.format(iterations))
    stats = hyperopt.optimize(benchmark, iterations)
    persistence.store_results(benchmark, stats)
    ls.append(stats)
    print('Finished after {}s'.format(stats.end - stats.start))
    print(stats)

# bohb
if config.bohb:
    print('Start bohb')
    bohb = BohbAdapter(config.timeout, config.n_jobs)
    stats = bohb.optimize(benchmark)
    persistence.store_results(benchmark, stats)
    ls.append(stats)
    print('Finished after {}s'.format(stats.end - stats.start))
    print(stats)

# RoBo
if config.robo:
    print('Start robo')
    robo = RoBoAdapter(config.timeout, config.n_jobs, config.seed)
    stats = robo.optimize(benchmark)
    persistence.store_results(benchmark, stats)
    ls.append(stats)
    print('Finished after {}s'.format(stats.end - stats.start))
    print(stats)

plot_results(benchmark, ls)
