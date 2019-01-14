import logging
from argparse import Namespace

import benchmark
from adapter.bohb import BohbAdapter
from adapter.grid_search import ObjectiveGridSearch
from adapter.hyperopt import HyperoptAdapter
from adapter.random_search import ObjectiveRandomSearch
from adapter.smac import SmacAdapter

logging.basicConfig(level=40)  # 10: debug; 20: info

config_dict = {
    'n_jobs': 5,
    'timeout': 5,
    'seed': 42,

    'random_search': True,
    'grid_search': True,
    'smac': True,
    'hyperopt': True,
    'bohb': True
}
config = Namespace(**config_dict)

benchmark = benchmark.Branin()

# Random Search
if config.random_search:
    rs = ObjectiveRandomSearch(config.timeout, config.n_jobs, random_state=config.seed)
    stats = rs.optimize(benchmark)
    print(stats.metadata)
    print(len(stats.evaluations))

# Grid Search
if config.grid_search:
    rs = ObjectiveGridSearch(config.timeout, config.n_jobs)
    stats = rs.optimize(benchmark)
    print(stats.metadata)
    print(len(stats.evaluations))

# SMAC
if config.smac:
    smac = SmacAdapter(config.timeout, config.n_jobs, config.seed)
    stats = smac.optimize(benchmark)
    print(stats.metadata)
    print(len(stats.evaluations))

# hyperopt
if config.hyperopt:
    hyperopt = HyperoptAdapter(config.timeout, config.n_jobs)
    stats = hyperopt.optimize(benchmark)
    print(stats.metadata)
    print(len(stats.evaluations))

# hyperopt
if config.bohb:
    bohb = BohbAdapter(config.timeout, config.n_jobs)
    stats = bohb.optimize(benchmark)
    print(stats.metadata)
    print(len(stats.evaluations))
