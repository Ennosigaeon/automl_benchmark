import math
import multiprocessing
import time

from hpolib.abstract_benchmark import AbstractBenchmark
from sklearn.model_selection import ParameterGrid

from adapter.base import OptimizationStatistic, EvaluationResult, BaseAdapter, OBJECTIVE_TIME_FACTOR
from config import GridSearchConverter


def query_objective_function(candidates: ParameterGrid, benchmark: AbstractBenchmark, timeout: float,
                             lock: multiprocessing.Lock, index: multiprocessing.Value, ):
    ls = []
    while timeout is None or time.time() < timeout:
        lock.acquire()
        i = index.value
        index.value += 1
        lock.release()

        try:
            config = candidates[i]
            # noinspection PyTypeChecker,PyArgumentList
            res = benchmark.objective_function(config)
            ls.append(EvaluationResult.from_dict(res, config))
        except IndexError:
            # Done
            break
    return ls


class ObjectiveGridSearch(BaseAdapter):
    def __init__(self, n_jobs: int, time_limit: float = None, iterations: int = None):
        super().__init__(n_jobs, time_limit, iterations)

        m = multiprocessing.Manager()
        self.lock = m.Lock()
        self.index = m.Value('i', 0)

    def estimate_grid_size(self, dimensions: int, objective_time: float = None) -> int:
        if self.time_limit is not None:
            t = objective_time * OBJECTIVE_TIME_FACTOR + 0.0005
            n = (self.time_limit / t) ** (1 / dimensions)
        else:
            n = self.iterations ** (1 / dimensions)

        return int(max(10, n))

    # noinspection PyMethodOverriding
    def optimize(self, benchmark: AbstractBenchmark, grid_size: int = 10):
        start = time.time()
        timeout = start + self.time_limit if self.time_limit else None

        statistics = OptimizationStatistic('Grid Search', start)

        # noinspection PyArgumentList
        config_space = benchmark.get_configuration_space(GridSearchConverter(n=grid_size))
        candidates = ParameterGrid(config_space)

        pool = multiprocessing.Pool(processes=self.n_jobs)
        for i in range(self.n_jobs):
            pool.apply_async(query_objective_function, args=(candidates, benchmark, timeout, self.lock, self.index),
                             callback=lambda res: statistics.add_result(res),
                             error_callback=self.log_async_error)

        pool.close()
        pool.join()
        statistics.stop_optimisation()

        return statistics
