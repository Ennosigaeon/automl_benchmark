from __future__ import division

import json
import os
import time
from typing import List, Union

import numpy as np
from atm.method import Method, HyperPartition
from btb.selection.selector import Selector
from btb.tuning import GP, GPEi, Uniform
from hpolib.abstract_benchmark import AbstractBenchmark
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor

from adapter.base import BaseAdapter, OptimizationStatistic, EvaluationResult
from config import BtbConverter


class BtbAdapter(BaseAdapter):

    def __init__(self, n_jobs: int, time_limit: float = None, iterations: int = None, seed: int = None):
        super().__init__(n_jobs, time_limit, iterations, seed)

    # noinspection PyMethodOverriding
    def optimize(self, benchmark: AbstractBenchmark) -> OptimizationStatistic:
        start = time.time()
        statistics = OptimizationStatistic('BTB', start)

        # noinspection PyArgumentList,PyTypeChecker
        methods = self._create_method(benchmark.get_configuration_space(BtbConverter()))

        hyperpartitions = []
        for method in methods:
            if len(method.name) > 0:
                for hp in method.get_hyperpartitions():
                    hp.categoricals.append(('algorithm', method.name))
                    hyperpartitions.append(hp)
            else:
                hyperpartitions += method.get_hyperpartitions()

        tuners = [FixedGP(hp.tunables, r_minimum=1) for hp in hyperpartitions]
        scores = {idx: tuner.y for idx, tuner in enumerate(tuners)}

        ls = []
        selector = Selector(scores.keys())
        for i in range(self.iterations):
            idx = selector.select(scores)

            params = tuners[idx].propose()
            params = self.__get_configuration_dict(hyperpartitions[idx], params)
            res = benchmark.objective_function(params)
            score = -1 * res['function_value']
            tuners[idx].add(params, score)

            res['config'] = params
            # print(res)
            ls.append(EvaluationResult.from_dict(res, params))

        statistics.add_result(ls)
        statistics.stop_optimisation()

        return statistics

    @staticmethod
    def __get_configuration_dict(hyperpartition: HyperPartition, tunables: dict) -> dict:
        tunables.update(dict(hyperpartition.constants))
        tunables.update(dict(hyperpartition.categoricals))
        return tunables

    @staticmethod
    def _create_method(conf: Union[dict, list]) -> List[Method]:
        if isinstance(conf, dict):
            conf = [conf]

        ls = []
        for c in conf:
            name = '/tmp/{}'.format(time.time())
            with open(name, 'w') as f:
                json.dump(c, f)
            ls.append(Method(name))
            os.remove(name)

        return ls


class FixedGP(GPEi):
    # predict and _acquire can be removed if inherited from GP

    def predict(self, X):
        if self.X.shape[0] < self.r_minimum:
            y = Uniform(self.tunables).predict(X)
            stdev = np.ones(len(y), dtype=np.float64)
        else:
            y, stdev = self.gp.predict(X, return_std=True)
        return np.array(list(zip(y, stdev)))

    def _acquire(self, predictions):
        Phi = norm.cdf
        N = norm.pdf

        mu, sigma = predictions.T
        if len(self.y) == 0:
            y_best = max(mu)
        else:
            y_best = max(self.y)

        # because we are maximizing the scores, we do mu-y_best rather than the inverse, as is
        # shown in most reference materials
        z = ((mu - y_best) / sigma).astype(np.float64)

        ei = sigma * (z * Phi(z) + N(z))

        return np.argmax(ei)

    def fit(self, X, y):
        """ Use X and y to train a Gaussian process. """
        super(GP, self).fit(X, y)

        # skip training the process if there aren't enough samples
        if X.shape[0] < self.r_minimum:
            return

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.gp = GaussianProcessRegressor(normalize_y=True)
        self.gp.fit(X, y)
