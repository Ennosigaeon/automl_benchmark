import time

import numpy as np
from hpolib.abstract_benchmark import AbstractBenchmark
from hyperopt import Trials, fmin, STATUS_FAIL, STATUS_OK, tpe

import util.logger
from adapter.base import BaseAdapter, OptimizationStatistic, EvaluationResult, OBJECTIVE_TIME_FACTOR
from config import HyperoptConverter

logger = util.logger.get()


class HyperoptAdapter(BaseAdapter):
    def __init__(self, n_jobs: int, time_limit: float = None, iterations: int = None, objective_time: float = None,
                 seed: int = None):
        super().__init__(n_jobs, time_limit, iterations, seed)
        self.timeout = None
        self.benchmark = None

        if iterations is None:
            if objective_time is None:
                raise ValueError('Unable to estimate number of iterations without objective time')
            self.iterations = self.estimate_iterations(objective_time)
            logger.debug('Using maximal {} iterations'.format(self.iterations))

    def estimate_iterations(self, objective_time: float) -> int:
        t = 1 / (objective_time * OBJECTIVE_TIME_FACTOR + 0.04)
        return int(self.time_limit * t)

    # noinspection PyMethodOverriding
    def optimize(self, benchmark: AbstractBenchmark) -> OptimizationStatistic:
        start = time.time()
        self.timeout = start + self.time_limit if self.time_limit else None
        self.benchmark = benchmark

        statistics = OptimizationStatistic('hyperopt', start)

        # noinspection PyArgumentList
        conf = benchmark.get_configuration_space(HyperoptConverter(as_scope=False))
        random_state = np.random.RandomState(self.seed) if self.seed is not None else None

        trials = Trials()
        # trials = MongoTrials('mongo://10.0.2.2:27017/hyperopt/jobs', exp_key='exp1')
        best = fmin(self.query_objective_function,
                    space=conf,
                    algo=tpe.suggest,
                    max_evals=self.iterations,
                    rstate=random_state,
                    trials=trials)

        ls = []
        for res in trials.results:
            if res['status'] == 'fail':
                if res['status_fail'] == 'Timeout reached':
                    break
                else:
                    logger.error('Unexpected error: {}'.format(res['status_fail']))
            ls.append(EvaluationResult(res['start'], res['end'], res['loss'], best))
        statistics.add_result(ls)
        statistics.stop_optimisation()

        return statistics

    def query_objective_function(self, conf):
        if (self.timeout is not None and time.time() > self.timeout):
            return {
                'status': STATUS_FAIL,
                'status_fail': 'Timeout reached'
            }

        res = self.benchmark.objective_function(conf)
        res['status'] = STATUS_OK
        res['loss'] = res['function_value']
        return res
