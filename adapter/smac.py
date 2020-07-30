import time
from typing import Dict, List

import numpy as np
from ConfigSpace import Configuration
from hpolib.abstract_benchmark import AbstractBenchmark
from smac.facade.smac_facade import SMAC
from smac.runhistory.runhistory import RunKey, RunValue
from smac.scenario.scenario import Scenario

from adapter.base import OptimizationStatistic, EvaluationResult, BaseAdapter
from config import ConfigSpaceConverter
from util import multiprocessor


def query_objective_function(benchmark: AbstractBenchmark, idx: int, seed: int,
                             time_limit: float = None, iterations=None):
    # noinspection PyArgumentList
    cs = benchmark.get_configuration_space(ConfigSpaceConverter())
    name = benchmark.get_meta_information()['name']
    random_state = np.random.RandomState(seed)

    scenario = {
        'abort_on_first_run_crash': True,
        'run_obj': 'quality',
        'deterministic': True,
        'shared-model': True,

        # 'cutoff_time': 10,
        'cs': cs,
        'initial_incumbent': 'RANDOM',

        'input_psmac_dirs': '/tmp/smac/{:s}/in/'.format(name),
        'output_dir': '/tmp/smac/{:s}/out/{:d}/{:d}'.format(name, int(time.time()), idx)
    }

    if time_limit is not None:
        scenario['wallclock_limit'] = time_limit
    else:
        scenario['runcount_limit'] = iterations

    def objective_function(configuration, **kwargs):
        d = {}
        algorithm = configuration._values.get('__choice__', '')
        if len(algorithm) > 0:
            n = len(algorithm) + 1
            d['algorithm'] = algorithm
        else:
            n = 0

        for key, value in configuration._values.items():
            if key == '__choice__':
                continue
            d[key[n:]] = value
        return benchmark.objective_function(d, **kwargs)['function_value']

    smac = SMAC(scenario=Scenario(scenario), tae_runner=objective_function, rng=random_state)
    x_star = smac.optimize()

    return smac.runhistory.data, x_star


class SmacAdapter(BaseAdapter):
    def __init__(self, n_jobs: int, time_limit: float = None, iterations: int = None, seed: int = None):
        super().__init__(n_jobs, time_limit, iterations, seed)

        if self.seed is None:
            raise ValueError('seed is required for smac')

    # noinspection PyMethodOverriding
    def optimize(self, benchmark: AbstractBenchmark, mean_objective_time: float = 0.1):
        start = time.time()
        statistics = OptimizationStatistic('SMAC', start)

        pool = multiprocessor.NoDaemonPool(processes=self.n_jobs)
        for i in range(self.n_jobs):
            pool.apply_async(query_objective_function,
                             args=(benchmark, i, self.seed + i, self.time_limit, self.iterations / self.n_jobs),
                             callback=lambda res: statistics.add_result(
                                 self._transform_result(res[0], res[1], start, mean_objective_time)
                             ),
                             error_callback=self.log_async_error)
        pool.close()
        pool.join()
        statistics.stop_optimisation()

        return statistics

    @staticmethod
    def _transform_result(history: Dict[RunKey, RunValue], best: Configuration,
                          start: float, mean_objective_time: float) -> List:
        end = time.time()
        n = len(history.values())

        # Exact overhead is not known. Mean and standard deviation empirically computed and now faked
        total = end - start - n * mean_objective_time
        overhead = np.random.normal(0.02149568831957658, 0.002992145452598064, n)
        overhead = (overhead / overhead.sum()) * total

        res = []
        for i, run_value in enumerate(history.values()):
            t = start + mean_objective_time * i + np.cumsum(overhead)[i]

            res.append(EvaluationResult(t, t + mean_objective_time, run_value.cost, best.get_dictionary()))
        return res
