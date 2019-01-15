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

        "input_psmac_dirs": "/tmp/smac/{:s}/in/".format(name),
        "output_dir": "/tmp/smac/{:s}/out/{:d}".format(name, random_state)
    })

    smac = SMAC(scenario=scenario, tae_runner=benchmark, rng=np.random.RandomState(random_state))
    x_star = smac.optimize()

    return smac.runhistory.data, x_star


class SmacAdapter(BaseAdapter):
    def __init__(self, time_limit: float, n_jobs: int, random_state: int = None):
        super().__init__(time_limit, n_jobs, random_state)

    # noinspection PyMethodOverriding
    def optimize(self, benchmark: AbstractBenchmark):
        start = time.time()
        statistics = OptimizationStatistic('SMAC', start, self.n_jobs)

        pool = multiprocessor.NoDaemonPool(processes=self.n_jobs)
        for i in range(self.n_jobs):
            rs = None if self.random_state is None else self.random_state + i
            pool.apply_async(query_objective_function, args=(benchmark, self.time_limit, rs),
                             callback=lambda res: statistics.add_result(self._transform_result(res[0], res[1], start)),
                             error_callback=self.log_async_error)
        pool.close()
        pool.join()
        statistics.stop_optimisation()

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
