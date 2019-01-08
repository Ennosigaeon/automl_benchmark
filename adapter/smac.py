import time
from typing import Dict, List

import numpy as np
from ConfigSpace import Configuration
from hpolib.abstract_benchmark import AbstractBenchmark
from smac.facade.smac_facade import SMAC
from smac.runhistory.runhistory import RunKey, RunValue
from smac.scenario.scenario import Scenario

from adapter.base import OptimizationStatistic, EvaluationResult, log_async_error
from config import ConfigSpaceConverter
from util import multiprocessor


def query_objective_function(benchmark: AbstractBenchmark, time_limit: float, random_state: int):
    # noinspection PyArgumentList
    cs = benchmark.get_configuration_space(ConfigSpaceConverter())
    name = benchmark.get_meta_information()['name']

    scenario = Scenario({
        'abort_on_first_run_crash': True,
        'run_obj': 'quality',
        'deterministic': True,
        'shared-model': True,

        "wallclock_limit": time_limit,
        # 'cutoff_time': 10,
        "cs": cs,

        "input_psmac_dirs": "./smac/{:s}/in/".format(name),
        "output_dir": "./smac/{:s}/out/{:d}".format(name, random_state)
    })

    smac = SMAC(scenario=scenario, tae_runner=benchmark, rng=np.random.RandomState(random_state))
    x_star = smac.optimize()

    return smac.runhistory.data, x_star


class SmacAdapter:
    def __init__(self, time_limit: float, n_jobs: int, random_state: int = None):
        self.time_limit = time_limit
        self.n_jobs = n_jobs
        self.random_state = random_state

    def optimize(self, benchmark: AbstractBenchmark):
        start = time.time()
        statistics = OptimizationStatistic('Grid Search', start)

        pool = multiprocessor.NoDaemonPool(processes=self.n_jobs)
        for i in range(self.n_jobs):
            rs = None if self.random_state is None else self.random_state + i
            pool.apply_async(query_objective_function, args=(benchmark, self.time_limit, rs),
                             callback=lambda res: statistics.add_result(self._transform_result(res[0], res[1], start)),
                             error_callback=log_async_error)
        pool.close()
        pool.join()

        return statistics

    @staticmethod
    def _transform_result(history: Dict[RunKey, RunValue], best: Configuration, start: float) -> List:
        res = []
        offset = 0
        for run_value in history.values():
            res.append(EvaluationResult(start + offset, start + offset + run_value.time,
                                        run_value.cost, best.get_dictionary()))
            offset += run_value.time
        return res
