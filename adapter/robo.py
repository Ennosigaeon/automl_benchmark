import time

from hpolib.abstract_benchmark import AbstractBenchmark
from robo.fmin import bayesian_optimization

from adapter.base import BaseAdapter, OptimizationStatistic, EvaluationResult
from config import RoBoConverter


class RoBoAdapter(BaseAdapter):

    def __init__(self, n_jobs: int, time_limit: float = None, iterations: int = None):
        super().__init__(n_jobs, time_limit, iterations)
        self.benchmark = None

        if self.iterations is None:
            raise NotImplementedError('Timeout not supported yet')

    # noinspection PyMethodOverriding
    def optimize(self, benchmark: AbstractBenchmark, model_type: str = 'gp_mcmc'):
        self.benchmark = benchmark

        start = time.time()
        statistics = OptimizationStatistic('RoBo {}'.format(model_type), start)
        # noinspection PyArgumentList
        lower, upper, names = benchmark.get_configuration_space(RoBoConverter())

        res = bayesian_optimization(lambda x: benchmark.objective_function(x)['function_value'],
                                    lower, upper,
                                    model_type=model_type,
                                    num_iterations=self.iterations)
        # res = fabolas(self._objective_function, lower, upper, num_iterations=4, s_min=0, s_max=0)

        ls = []
        previous = start
        for i in range(len(res['runtime'])):
            d = {}
            for j in range(len(names)):
                d[names[j]] = res['incumbents'][i][j]

            begin = previous + res['overhead'][i]
            ls.append(
                EvaluationResult(begin, start + res['runtime'][i], res['incumbent_values'][i], d)
            )
            previous = start + res['runtime'][i]

        statistics.add_result(ls)
        statistics.stop_optimisation()

        return statistics
