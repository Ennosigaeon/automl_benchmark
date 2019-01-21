import abc
import time
import traceback
from typing import List, Union

import numpy as np
from hpolib.abstract_benchmark import AbstractBenchmark

OBJECTIVE_TIME_FACTOR = 1.5


class EvaluationResult:

    def __init__(self, start: float, end: float, score: float, config: dict):
        self.start = start
        self.end = end
        self.score = score
        self.config = config

    def __str__(self):
        return str(self.as_dict())

    def __repr__(self):
        return str(self)

    def as_dict(self):
        return {
            'start': self.start,
            'end': self.end,
            'score': self.score,
            'config': self.config
        }

    @staticmethod
    def from_dict(d: dict, conf: dict) -> 'EvaluationResult':
        return EvaluationResult(d['start'], d['end'], d['function_value'], conf)


class OptimizationStatistic:

    def __init__(self, algorithm: str, start: float, n_jobs: int):
        self.algorithm = algorithm
        self.n_jobs = n_jobs

        self.start = start
        self.end = None

        self.iterations = 0
        self.score = None
        self.best = None
        self.runtime = {}

        self.evaluations: List[EvaluationResult] = []

    def add_result(self, result: List[EvaluationResult]):
        self.evaluations.extend(result)

    def stop_optimisation(self):
        self.end = time.time()
        self.evaluations = sorted(self.evaluations, key=lambda ev: ev.end)
        self.iterations = len(self.evaluations)

        best_score = float("inf")
        best = None
        for ev in self.evaluations:
            if ev.score < best_score:
                best_score = ev.score
                best = ev.config
        self.score = best_score
        self.best = best

        total = self.end - self.start
        objective_function = np.array([ev.end - ev.start for ev in self.evaluations])

        overhead = []
        previous = self.start
        for ev in self.evaluations:
            overhead.append(ev.start - previous)
            previous = ev.end
        overhead.append(self.end - previous)
        overhead = np.array(overhead)

        self.runtime = {
            'total': total,
            'objective_function': [objective_function.mean(), objective_function.var(), objective_function.sum()],
            'overhead': [overhead.mean(), overhead.var(), overhead.sum()]
        }

    @property
    def incumbents(self):
        ls = []

        current_best = float("inf")
        for ev in self.evaluations:
            if ev.score < current_best:
                ls.append(ev)
                current_best = ev.score
        return ls

    def as_numpy(self, incumbent: bool = True, x_axis: str = 'iterations'):
        """
        Returns the evaluations as two one-dimensional numpy arrays.
        :param incumbent: Only include improvements and not all runs
        :param x_axis: Specifies the type of the x-axis. Can be either 'iterations' or 'time'
        :return: x, y array
        """
        x = []
        y = []

        if x_axis == 'time':
            ls = self.incumbents if incumbent else self.evaluations
            for ev in ls:
                x.append(ev.end - self.start)
                y.append(ev.score)
        elif x_axis == 'iterations':
            current_best = float("inf")
            for idx, ev in enumerate(self.evaluations):
                if self.incumbents and ev.score > current_best:
                    continue
                x.append(idx)
                y.append(ev.score)
        else:
            raise ValueError('Unknown x_axis {}'.format(x_axis))

        return np.array(x), np.array(y)

    def as_dict(self, include_evaluations=False):
        ls = self.incumbents if include_evaluations else []
        return {
            'algorithm': self.algorithm,
            'n_jobs': self.n_jobs,

            'start': self.start,
            'end': self.end,

            'iterations': self.iterations,
            'score': self.score,
            'best': self.best,
            'runtime': self.runtime,
            'incumbents': [ev.as_dict() for ev in ls]
        }

    @staticmethod
    def from_dict(d: dict) -> 'OptimizationStatistic':
        instance = OptimizationStatistic(d['algorithm'], d['start'], d['n_jobs'])
        instance.end = d['end']
        instance.iterations = d['iterations']
        instance.score = d['score']
        instance.best = d['best']
        instance.runtime = d['runtime']
        instance.evaluations = [EvaluationResult(**f) for f in d['incumbents']]
        return instance

    def __str__(self):
        return str(self.as_dict(include_evaluations=False))


class BaseAdapter(abc.ABC):

    @staticmethod
    def log_async_error(ex: Exception):
        traceback.print_exception(type(ex), ex, None)

    def __init__(self, n_jobs: int, time_limit: float = None, iterations: int = None,
                 random_state: Union[None, int] = None):
        self.n_jobs = n_jobs
        self.time_limit = time_limit
        self.iterations = iterations
        self.random_state = random_state

        if time_limit is None and iterations is None:
            raise ValueError('Expecting limited runtime or limited number of iterations')

    @abc.abstractmethod
    def optimize(self, benchmark: AbstractBenchmark, **kwargs) -> OptimizationStatistic:
        pass
