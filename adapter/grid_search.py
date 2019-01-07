import multiprocessing
import time

from hpolib.abstract_benchmark import AbstractBenchmark
from sklearn.model_selection import ParameterGrid

from adapter.base import OptimizationStatistic, EvaluationResult
from config import GridSearchConverter


def query_objective_function(conf: dict, benchmark: AbstractBenchmark, time_limit: float):
    start = time.time()
    if start >= time_limit:
        raise TimeoutError()

    # noinspection PyTypeChecker,PyArgumentList
    score = benchmark.objective_function(conf)
    end = time.time()
    return [EvaluationResult(start, end, score['function_value'], conf)]


class ObjectiveGridSearch:
    def __init__(self, time_limit: float, n_jobs: int):
        self.time_limit = time_limit
        self.n_jobs = n_jobs

    def optimize(self, benchmark: AbstractBenchmark):
        start = time.time()
        timeout = start + self.time_limit
        statistics = OptimizationStatistic('Grid Search', start)

        pool = multiprocessing.Pool(processes=self.n_jobs)

        # noinspection PyArgumentList
        config_space = benchmark.get_configuration_space(GridSearchConverter())
        candidates = ParameterGrid(config_space)
        for config in candidates:
            pool.apply_async(query_objective_function, args=(config, benchmark, timeout),
                             callback=lambda res: statistics.add_result(res))

        pool.close()
        pool.join()

        return statistics
