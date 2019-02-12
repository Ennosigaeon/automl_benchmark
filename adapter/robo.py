import math
import time

import numpy as np
from hpolib.abstract_benchmark import AbstractBenchmark
from robo.fmin import bayesian_optimization

from adapter.base import BaseAdapter, OptimizationStatistic, EvaluationResult
from config import RoBoConverter, MetaConfigCollection, UNI_FLOAT, UNI_INT, CATEGORICAL


class RoBoAdapter(BaseAdapter):

    def __init__(self, n_jobs: int, time_limit: float = None, iterations: int = None, seed: int = None):
        super().__init__(n_jobs, time_limit, iterations, seed)
        self.benchmark = None

        if self.iterations is None:
            raise NotImplementedError('Timeout not supported yet')

    # noinspection PyMethodOverriding
    def optimize(self, benchmark: AbstractBenchmark, model_type: str = 'gp_mcmc'):
        self.benchmark = benchmark

        start = time.time()
        statistics = OptimizationStatistic('RoBo {}'.format(model_type), start)

        ls = []
        if benchmark.get_meta_information().get('cash', False):
            # noinspection PyArgumentList
            for key, value in benchmark.get_configuration_space(RoBoConverter()).items():
                ls.append({
                    'lower': value[0],
                    'upper': value[1],
                    'names': value[2],
                    'algorithm': key
                })
        else:
            # noinspection PyArgumentList
            lower, upper, names = benchmark.get_configuration_space(RoBoConverter())
            ls.append({
                'lower': lower,
                'upper': upper,
                'names': names
            })

        random_state = np.random.RandomState(self.seed) if self.seed is not None else None
        cs = benchmark.get_configuration_space()

        n = max(3, math.ceil(self.iterations / len(ls)))
        res = []
        for config in ls:
            tmp = bayesian_optimization(lambda x: self.objective_function(x, config, cs),
                                        config['lower'], config['upper'],
                                        model_type=model_type,
                                        num_iterations=n,
                                        rng=random_state)
            res.append(tmp)

        result = []
        previous = start
        for idx, r in enumerate(res):
            for i in range(len(r['runtime'])):
                d = {}
                for j, name in enumerate(ls[idx]['names']):
                    d[name] = r['incumbents'][i][j]

                begin = previous + r['overhead'][i]
                result.append(
                    EvaluationResult(begin, start + r['runtime'][i], r['incumbent_values'][i], d)
                )
                previous = start + r['runtime'][i]

        statistics.add_result(result)
        statistics.stop_optimisation()

        return statistics

    def objective_function(self, x, config: dict, config_space: MetaConfigCollection):
        if 'algorithm' not in config:
            res = self.benchmark.objective_function(x)
        else:
            d = {
                'algorithm': config['algorithm']
            }
            cs = config_space.algos[d['algorithm']]
            for idx, name in enumerate(config['names']):
                feature = cs.dict[name]
                if feature.type == UNI_FLOAT:
                    d[name] = x[idx]
                else:
                    x_int = int(round(x[idx]))
                    if feature.type == UNI_INT:
                        d[name] = x_int
                    elif feature.type == CATEGORICAL:
                        d[name] = feature.choices[x_int]
                    else:
                        raise ValueError('Unknown type {}'.format(feature.type))
            res = self.benchmark.objective_function(d)

        return res['function_value']
