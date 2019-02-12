import multiprocessing
import random
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


def timed_query(benchmark: AbstractBenchmark, timeout: float, seed: int):
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
        cs = benchmark.get_configuration_space(RandomSearchConverter())
        if benchmark.get_meta_information().get('cash', False):
            key, value = random.sample(list(cs.items()), 1)[0]
            cs = value.copy()
            cs['algorithm'] = [key]
        conf = list(CustomParameterSampler(cs, 1, random_state=random_state))[0]
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
