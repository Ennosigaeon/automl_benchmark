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
persistence.clear_old_results(benchmark)

# Random Search
if config.random_search:
    rs = ObjectiveRandomSearch(config.timeout, config.n_jobs)
    stats = rs.optimize(benchmark)
    persistence.store_results(benchmark, stats)
    ls.append(stats)
    print(stats)

# Grid Search
if config.grid_search:
    rs = ObjectiveGridSearch(config.timeout, config.n_jobs)
    stats = rs.optimize(benchmark)
    persistence.store_results(benchmark, stats)
    ls.append(stats)
    print(stats)

# SMAC
if config.smac:
    smac = SmacAdapter(config.timeout, config.n_jobs, config.seed)
    stats = smac.optimize(benchmark)
    persistence.store_results(benchmark, stats)
    ls.append(stats)
    print(stats)

# hyperopt
if config.hyperopt:
    hyperopt = HyperoptAdapter(config.timeout, config.n_jobs)
    stats = hyperopt.optimize(benchmark)
    persistence.store_results(benchmark, stats)
    ls.append(stats)
    print(stats)

# bohb
if config.bohb:
    bohb = BohbAdapter(config.timeout, config.n_jobs)
    stats = bohb.optimize(benchmark)
    persistence.store_results(benchmark, stats)
    ls.append(stats)
    print(stats)

# RoBo
if config.robo:
    robo = RoBoAdapter(config.timeout, config.n_jobs, config.seed)
    stats = robo.optimize(benchmark)
    persistence.store_results(benchmark, stats)
    ls.append(stats)
    print(stats)

plot_results(ls)
