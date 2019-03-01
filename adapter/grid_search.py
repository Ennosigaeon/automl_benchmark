import math
import multiprocessing
import time

import numpy as np
from hpolib.abstract_benchmark import AbstractBenchmark
from sklearn.model_selection import ParameterGrid

from adapter.base import OptimizationStatistic, EvaluationResult, BaseAdapter, OBJECTIVE_TIME_FACTOR
from config import GridSearchConverter
from util.multiprocessor import NoDaemonPool


def query_objective_function(candidates: ParameterGrid, benchmark: AbstractBenchmark, iterations: int, timeout: float,
                             lock: multiprocessing.Lock, index: multiprocessing.Value):
    ls = []
    idx = 0
    while timeout is None or time.time() < timeout:
        lock.acquire()
        i = index.value
        index.value += 1
        lock.release()

        if iterations is not None and i >= iterations:
            break

        try:
            config = candidates[idx]
            for key, value in config.items():
                if isinstance(value, np.int64) or isinstance(value, np.float64):
                    config[key] = value.item()

            # noinspection PyTypeChecker,PyArgumentList
            res = benchmark.objective_function(config)
            ls.append(EvaluationResult.from_dict(res, config))
            idx += 1
        except IndexError:
            # Done
            lock.acquire()
            index.value -= 1
            lock.release()
            break
    return ls


class ObjectiveGridSearch(BaseAdapter):
    def __init__(self, n_jobs: int, time_limit: float = None, iterations: int = None):
        super().__init__(n_jobs, time_limit, iterations)

        m = multiprocessing.Manager()
        self.lock = m.Lock()
        self.index = m.Value('i', 0)

    def estimate_grid_size(self, dimensions: int = 0, objective_time: float = None) -> int:
        if self.time_limit is not None:
            t = objective_time * OBJECTIVE_TIME_FACTOR + 0.0005
            n = (self.time_limit / t) ** (1 / dimensions)
        elif dimensions != 0:
            n = math.ceil(self.iterations ** (1 / dimensions))
        else:
            n = 5

        return int(max(1, n))

    # noinspection PyMethodOverriding
    def optimize(self, benchmark: AbstractBenchmark, grid_size: int = 10):
        start = time.time()
        timeout = start + self.time_limit if self.time_limit else None

        statistics = OptimizationStatistic('Grid Search', start)

        # noinspection PyArgumentList
        config_space = benchmark.get_configuration_space(GridSearchConverter(n=grid_size))
        candidate_list = []
        if benchmark.get_meta_information().get('cash', False):
            for key, value in config_space.items():
                conf = value.copy()
                conf['algorithm'] = [key]
                candidate_list.append(ParameterGrid(conf))
        else:
            candidate_list.append(ParameterGrid(config_space))

        pool = NoDaemonPool(processes=self.n_jobs)
        for candidates in candidate_list:
            for i in range(self.n_jobs):
                pool.apply_async(query_objective_function,
                                 args=(candidates, benchmark, self.iterations, timeout, self.lock, self.index),
                                 callback=lambda res: statistics.add_result(res),
                                 error_callback=self.log_async_error)

        pool.close()
        pool.join()
        statistics.stop_optimisation()

        return statistics
