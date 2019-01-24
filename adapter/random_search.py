import multiprocessing
import time

import numpy as np
from hpolib.abstract_benchmark import AbstractBenchmark
from sklearn.model_selection import ParameterSampler
from sklearn.utils import check_random_state

from adapter.base import OptimizationStatistic, EvaluationResult, BaseAdapter
from config import RandomSearchConverter


class CustomParameterSampler(ParameterSampler):
    def __iter__(self):
        sample = ParameterSampler.__iter__(self)
        rnd = check_random_state(self.random_state)
        for s in sample:
            for k, v in s.items():
                if hasattr(v, "rvs"):
                    s[k] = v.rvs(random_state=rnd)
            yield s


def timed_query(benchmark: AbstractBenchmark, timeout: float, seed:int):
    random_state = np.random.RandomState(seed)
    ls = []
    while time.time() < timeout:
        # noinspection PyTypeChecker,PyArgumentList
        conf = list(CustomParameterSampler(benchmark.get_configuration_space(RandomSearchConverter()), 1,
                                           random_state=random_state))[0]
        res = benchmark.objective_function(conf)
        ls.append(EvaluationResult.from_dict(res, conf))
    return ls


def run_counted_query(benchmark: AbstractBenchmark, iterations: int, seed: int,
                      lock: multiprocessing.Lock, index: multiprocessing.Value, ):
    random_state = np.random.RandomState(seed)
    ls = []
    while True:
        lock.acquire()
        i = index.value
        index.value += 1
        lock.release()

        if i >= iterations:
            break

        # noinspection PyTypeChecker,PyArgumentList
        conf = list(CustomParameterSampler(benchmark.get_configuration_space(RandomSearchConverter()), 1,
                                           random_state=random_state))[0]
        res = benchmark.objective_function(conf)
        ls.append(EvaluationResult.from_dict(res, conf))
    return ls


class ObjectiveRandomSearch(BaseAdapter):
    def __init__(self, n_jobs: int, time_limit: float = None, iterations: int = None, seed: int = None):
        super().__init__(n_jobs, time_limit, iterations, seed)

        if self.seed is None:
            raise ValueError('seed is required for random search')

        m = multiprocessing.Manager()
        self.lock = m.Lock()
        self.index = m.Value('i', 0)

    # noinspection PyMethodOverriding
    def optimize(self, benchmark: AbstractBenchmark):
        start = time.time()
        statistics = OptimizationStatistic('Random Search', start)

        pool = multiprocessing.Pool(processes=self.n_jobs)
        for i in range(self.n_jobs):
            if self.time_limit is not None:
                timeout = start + self.time_limit
                pool.apply_async(timed_query, args=(benchmark, timeout, self.seed + i),
                                 callback=lambda res: statistics.add_result(res),
                                 error_callback=self.log_async_error)
            else:
                pool.apply_async(run_counted_query,
                                 args=(benchmark, self.iterations, self.seed + i, self.lock, self.index),
                                 callback=lambda res: statistics.add_result(res),
                                 error_callback=self.log_async_error)

        pool.close()
        pool.join()
        statistics.stop_optimisation()

        return statistics

# class CustomRandomSearch(RandomizedSearchCV):
#
#     def _run_search(self, evaluate_candidates):
#         """Search n_iter candidates from param_distributions"""
#         evaluate_candidates(
#             CustomParameterSampler(self.param_distributions, self.n_iter, random_state=self.random_state)
#         )
#
#
# class CashRandomSearch:
#     def __init__(self, config_space: dict, time_limit: float, scoring: str, n_jobs: int, random_state=None):
#         self.config_space = config_space
#         self.time_limit = time_limit
#         self.scoring = scoring
#         self.n_jobs = n_jobs
#         self.random_state = random_state
#
#     def fit(self, X, y):
#         start = time.time()
#         limit = start + self.time_limit
#         statistics = OptimizationStatistic('Random Search', start)
#
#         scoring = self.scoring
#
#         def test_configurations(i: int, configuration: dict, time_limit: float):
#             while time.time() < time_limit:
#                 estimator_name = random.choice(configuration.keys())
#                 config_space = configuration[estimator_name]
#
#                 # Why is this necessary?
#                 if isinstance(config_space, list):
#                     config_space = random.choice(config_space)
#
#                 estimator = self.__create_estimator(estimator_name)
#                 random_search = CustomRandomSearch(estimator, config_space, n_iter=1, n_jobs=1, cv=10, refit=False,
#                                                    error_score=np.nan, scoring=scoring)
#                 try:
#                     random_search.fit(X, y)
#                     now = time.time()
#                     best_score = random_search.best_score_
#                     best_estimator = random_search.best_estimator_
#
#                     # statistics.add_result(EvaluationResult(now, time.time(), best_score, best_estimator))
#                 except ValueError:
#                     # TODO log failure
#                     pass
#
#         pool = multiprocessing.Pool(processes=self.n_jobs)
#         for i in range(self.n_jobs):
#             pool.apply(test_configurations, args=(i, self.config_space, limit))
#         pool.close()
#         pool.join()
#
#         return statistics
#
#     @staticmethod
#     def __create_estimator(name):
#         module_name = name.rpartition(".")[0]
#         class_name = name.split(".")[-1]
#         module = import_module(module_name)
#         class_ = getattr(module, class_name)
#         return class_()
