import logging
import time
from argparse import Namespace

import benchmark
from adapter.bohb import BohbAdapter
from adapter.grid_search import ObjectiveGridSearch
from adapter.hyperopt import HyperoptAdapter
from adapter.optunity_adapter import OptunityAdapter
from adapter.random_search import ObjectiveRandomSearch
from adapter.robo import RoBoAdapter
from adapter.smac import SmacAdapter
from evaluation.base import MongoPersistent
from evaluation.visualization import plot_results

logging.basicConfig(level=40)  # 10: debug; 20: info
persistence = MongoPersistent('10.0.2.2')

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
    'robo': False,
    'optunity': True
}
config = Namespace(**config_dict)

benchmark = benchmark.Branin()
ls = []
# persistence.clear_old_results(benchmark)

objective_time = None

# Random Search
if config.random_search:
    print('Start random search')
    rs = ObjectiveRandomSearch(config.n_jobs, config.timeout, config.iterations, config.seed)
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
    gs = ObjectiveGridSearch(config.n_jobs, config.timeout, config.iterations)
    n = gs.estimate_grid_size(len(benchmark.get_meta_information()['bounds']), objective_time)
    print('Using grid size of {}'.format(n))
    stats = gs.optimize(benchmark, n)
    persistence.store_results(benchmark, stats)
    ls.append(stats)
    print('Finished after {}s'.format(stats.end - stats.start))
    print(stats)

# SMAC
if config.smac:
    print('Start SMAC')
    smac = SmacAdapter(config.n_jobs, config.timeout, config.iterations, config.seed)
    stats = smac.optimize(benchmark)
    persistence.store_results(benchmark, stats)
    ls.append(stats)
    print('Finished after {}s'.format(stats.end - stats.start))
    print(stats)

# hyperopt
if config.hyperopt:
    print('Start hyperopt')
    hyperopt = HyperoptAdapter(config.n_jobs, config.timeout, config.iterations)
    stats = hyperopt.optimize(benchmark)
    persistence.store_results(benchmark, stats)
    ls.append(stats)
    print('Finished after {}s'.format(stats.end - stats.start))
    print(stats)

# bohb
if config.bohb:
    print('Start bohb')
    bohb = BohbAdapter(config.n_jobs, config.timeout, config.iterations)
    stats = bohb.optimize(benchmark)
    persistence.store_results(benchmark, stats)
    ls.append(stats)
    print('Finished after {}s'.format(stats.end - stats.start))
    print(stats)

# RoBo
if config.robo:
    print('Start robo')
    robo = RoBoAdapter(config.n_jobs, config.timeout, config.iterations)
    stats = robo.optimize(benchmark)
    persistence.store_results(benchmark, stats)
    ls.append(stats)
    print('Finished after {}s'.format(stats.end - stats.start))
    print(stats)

if config.optunity:
    print('Start optunity')
    optunity = OptunityAdapter(config.n_jobs, config.timeout, config.iterations)
    stats = optunity.optimize(benchmark)
    persistence.store_results(benchmark, stats)
    ls.append(stats)
    print('Finished after {}s'.format(stats.end - stats.start))
    print(stats)

plot_results(benchmark, ls)
