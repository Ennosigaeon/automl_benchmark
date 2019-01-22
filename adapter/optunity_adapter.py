import sys
import time

import optunity
from hpolib.abstract_benchmark import AbstractBenchmark
from optunity import search_spaces
from optunity.api import suggest_solver, make_solver, _wrap_hard_box_constraints, optimize
from optunity.functions import wraps, CallLog

from adapter.base import BaseAdapter, OptimizationStatistic, EvaluationResult
from config import OptunityConverter


def logged(f):
    if hasattr(f, 'call_log'):
        return f

    @wraps(f)
    def wrapped_f(*args, **kwargs):
        value = f(*args, **kwargs)
        wrapped_f.call_log.insert(value['function_value'], start=value['start'], end=value['end'], **kwargs)
        return value['function_value']

    wrapped_f.call_log = CallLog()
    return wrapped_f


class OptunityAdapter(BaseAdapter):

    def __init__(self, n_jobs: int, time_limit: float = None, iterations: int = None):
        super().__init__(n_jobs, time_limit, iterations)
        self.benchmark = None

        if self.iterations is None:
            raise NotImplementedError('Timeout not supported yet')

    def optimize(self, benchmark: AbstractBenchmark, **kwargs) -> OptimizationStatistic:
        start = time.time()
        self.benchmark = benchmark
        statistics = OptimizationStatistic('Optunity', start, self.n_jobs)

        # noinspection PyArgumentList
        conf = benchmark.get_configuration_space(OptunityConverter())

        tree = search_spaces.SearchTree(conf)
        box = tree.to_box()

        f = tree.wrap_decoder(self.objective_function)
        f = _wrap_hard_box_constraints(f, box, 10000)

        suggestion = suggest_solver(self.iterations, "particle swarm", **box)
        solver = make_solver(**suggestion)

        solution, details = optimize(solver, f, maximize=False, max_evals=self.iterations, decoder=tree.decode,
                                     pmap=optunity.parallel.pmap)

        ls = []
        for meta, value in f.call_log.data.items():
            d = meta._asdict()
            start = d.pop('start', None)
            end = d.pop('end', None)
            ls.append(EvaluationResult(start, end, value, d))

        statistics.add_result(ls)
        statistics.stop_optimisation()

        return statistics

    @logged
    def objective_function(self, **kwargs):
        return self.benchmark.objective_function(kwargs)
