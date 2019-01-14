import time

from hpolib.abstract_benchmark import AbstractBenchmark
from hyperopt import Trials, fmin, STATUS_FAIL, STATUS_OK, tpe

from adapter.base import BaseAdapter, OptimizationStatistic, EvaluationResult
from config import HyperoptConverter


class HyperoptAdapter(BaseAdapter):
    def __init__(self, time_limit: float, n_jobs: int):
        super().__init__(time_limit, n_jobs)
        self.timeout = None
        self.benchmark = None

    # noinspection PyMethodOverriding
    def optimize(self, benchmark: AbstractBenchmark) -> OptimizationStatistic:
        start = time.time()
        self.timeout = start + self.time_limit
        self.benchmark = benchmark

        statistics = OptimizationStatistic('hyperopt', start, self.n_jobs)

        # noinspection PyArgumentList
        conf = benchmark.get_configuration_space(HyperoptConverter(as_scope=False))

        trials = Trials()
        # trials = MongoTrials('mongo://10.0.2.2:27017/hyperopt/jobs', exp_key='exp1')
        best = fmin(self.query_objective_function,
                    space=conf,
                    algo=tpe.suggest,
                    max_evals=100,
                    rstate=self.random_state,
                    trials=trials)

        ls = []
        for res in trials.results:
            if res['status'] == 'fail':
                if res['status_fail'] == 'Timeout reached':
                    break
                else:
                    print('Unexpected error: {}'.format(res['status_fail']))
            ls.append(EvaluationResult(res['start'], res['end'], res['loss'], best))
        statistics.add_result(ls)
        statistics.stop_optimisation()

        return statistics

    def query_objective_function(self, conf):
        if (time.time() > self.timeout):
            return {
                'status': STATUS_FAIL,
                'status_fail': 'Timeout reached'
            }

        res = self.benchmark.objective_function(conf)
        res['status'] = STATUS_OK
        res['loss'] = res['function_value']
        return res
