import abc
import time
import traceback
import numpy as np
from typing import List, Union

from hpolib.abstract_benchmark import AbstractBenchmark


class EvaluationResult:

    def __init__(self, start_time: float, end_time: float, score: float, params: dict):
        self.start_time = start_time
        self.end_time = end_time
        self.score = score
        self.params = params

    def __str__(self):
        d = {
            'start': self.start_time,
            'end': self.end_time,
            'score': self.score,
            'params': self.params
        }
        return str(d)

    def __repr__(self):
        return str(self)


class OptimizationStatistic:

    def __init__(self, algorithm: str, start: float, n_jobs: int):
        self.metadata = {
            'algorithm': algorithm,
            'start': start,
            'end': None,
            'n_jobs': n_jobs,
            'runtime': {}
        }
        self.evaluations: List[EvaluationResult] = []

    def add_result(self, result: List[EvaluationResult]):
        self.evaluations.extend(result)

    def stop_optimisation(self):
        self.metadata['end'] = time.time()

        total = self.metadata['end'] - self.metadata['start']
        objective_function = np.array([ev.end_time - ev.start_time for ev in self.evaluations])

        overhead = []
        previous = self.metadata['start']
        for ev in self.evaluations:
            overhead.append(ev.start_time - previous)
            previous = ev.end_time
        overhead.append(self.metadata['end'] - previous)
        overhead = np.array(overhead)

        self.metadata['runtime'] = {
            'total': total,
            'objective_function': [objective_function.mean(), objective_function.var()],
            'overhead': [overhead.mean(), overhead.var()]
        }


class BaseAdapter(abc.ABC):

    @staticmethod
    def log_async_error(ex: Exception):
        traceback.print_exception(type(ex), ex, None)

    def __init__(self, time_limit: float, n_jobs: int, random_state: Union[None, int] = None):
        self.time_limit = time_limit
        self.n_jobs = n_jobs
        self.random_state = random_state

    @abc.abstractmethod
    def optimize(self, benchmark: AbstractBenchmark, **kwargs) -> OptimizationStatistic:
        pass
