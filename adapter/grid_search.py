import multiprocessing
import time

from hpolib.abstract_benchmark import AbstractBenchmark
from sklearn.model_selection import ParameterGrid

from adapter.base import OptimizationStatistic, EvaluationResult, BaseAdapter
from config import GridSearchConverter


def query_objective_function(candidates: list, benchmark: AbstractBenchmark, timeout: float,
                             lock: multiprocessing.Lock, index: multiprocessing.Value, ):
    res = []
    while time.time() < timeout:
        lock.acquire()
        i = index.value
        index.value += 1
        lock.release()

        try:
            config = candidates[i]
            start = time.time()

            # noinspection PyTypeChecker,PyArgumentList
            score = benchmark.objective_function(config)
            end = time.time()

            res.append(EvaluationResult(start, end, score['function_value'], config))
        except IndexError:
            # Done
            break
    return res


class ObjectiveGridSearch(BaseAdapter):
    def __init__(self, time_limit: float, n_jobs: int):
        super().__init__(time_limit, n_jobs)

        m = multiprocessing.Manager()
        self.lock = m.Lock()
        self.index = m.Value('i', 0)

    # noinspection PyMethodOverriding
    def optimize(self, benchmark: AbstractBenchmark):
        # noinspection PyArgumentList
        config_space = benchmark.get_configuration_space(GridSearchConverter())
        candidates = ParameterGrid(config_space)

        start = time.time()
        timeout = start + self.time_limit
        statistics = OptimizationStatistic('Grid Search', start)

        pool = multiprocessing.Pool(processes=self.n_jobs)
        for i in range(self.n_jobs):
            pool.apply_async(query_objective_function, args=(candidates, benchmark, timeout, self.lock, self.index),
                             callback=lambda res: statistics.add_result(res),
                             error_callback=self.log_async_error)

        pool.close()
        pool.join()

        return statistics
