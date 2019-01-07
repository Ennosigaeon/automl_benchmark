from argparse import Namespace

from adapter.grid_search import ObjectiveGridSearch
from adapter.random_search import ObjectiveRandomSearch
from adapter.smac import SmacAdapter
from benchmark import BraninFunction

config_dict = {
    'n_jobs': 10,
    'timeout': 30,
    'seed': 42,

    'random_search': False,
    'grid_search': False,
    'smac': True
}
config = Namespace(**config_dict)

b = BraninFunction()

# Random Search
if config.random_search:
    rs = ObjectiveRandomSearch(config.timeout, config.n_jobs, random_state=config.seed)
    stats = rs.optimize(b)
    print(stats.evaluations)

# Grid Search
if config.grid_search:
    rs = ObjectiveGridSearch(config.timeout, config.n_jobs)
    stats = rs.optimize(b)
    print(stats.evaluations)

# SMAC
if config.smac:
    smac = SmacAdapter(config.timeout, config.n_jobs, config.seed)
    stats = smac.optimize(b)
    print(stats.evaluations)
