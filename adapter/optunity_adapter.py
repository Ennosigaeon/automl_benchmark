import multiprocessing
import re
import time

from hpolib.abstract_benchmark import AbstractBenchmark
from optunity import search_spaces
from optunity.api import suggest_solver, make_solver, _wrap_hard_box_constraints, optimize
from optunity.functions import wraps, CallLog

from adapter.base import BaseAdapter, OptimizationStatistic, EvaluationResult
from config import OptunityConverter


def _fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        value = f(*x)
        if hasattr(f, 'call_log'):
            k = list(f.call_log.keys())[-1]
            q_out.put((i, value, k))
        else:
            q_out.put((i, value))


# http://stackoverflow.com/a/16071616
def pmap(f, *args, **kwargs):
    """Parallel map using multiprocessing.

    :param f: the callable
    :param args: arguments to f, as iterables
    :returns: a list containing the results

    .. warning::
        This function will not work in IPython: https://github.com/claesenm/optunity/issues/8.

    .. warning::
        Python's multiprocessing library is incompatible with Jython.

    """
    nprocs = kwargs.get('number_of_processes', multiprocessing.cpu_count())
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=_fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = False
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(zip(*args))]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]
    [p.join() for p in proc]

    # FIXME: strong coupling between pmap and functions.logged
    if hasattr(f, 'call_log'):
        for _, value, k in sorted(res):
            f.call_log[k] = value
        return [x for i, x, _ in sorted(res)]
    else:
        return [x for i, x in sorted(res)]


def create_pmap(number_of_processes):
    def pmap_bound(f, *args):
        return pmap(f, *args, number_of_processes=number_of_processes)

    return pmap_bound


def logged(f):
    if hasattr(f, 'call_log'):
        return f

    @wraps(f)
    def wrapped_f(*args, **kwargs):
        config = {k: v for k, v in kwargs.items() if v is not None and k != ''}
        for key, value in config.items():
            if isinstance(value, str):
                if re.match('^\\d+$', value) is not None:
                    config[key] = int(value)
                elif value == 'True' or value == 'False':
                    config[key] = bool(value)

        value = f(*args, **config)
        wrapped_f.call_log.insert(value['function_value'], start=value['start'], end=value['end'], **config)
        return value['function_value']

    wrapped_f.call_log = CallLog()
    return wrapped_f


class OptunityAdapter(BaseAdapter):

    def __init__(self, n_jobs: int, time_limit: float = None, iterations: int = None, seed: int = None):
        super().__init__(n_jobs, time_limit, iterations, seed)
        self.benchmark = None

        if self.iterations is None:
            raise NotImplementedError('Timeout not supported yet')

    def optimize(self, benchmark: AbstractBenchmark, **kwargs) -> OptimizationStatistic:
        start = time.time()
        self.benchmark = benchmark
        statistics = OptimizationStatistic('Optunity', start)

        # noinspection PyArgumentList
        conf = benchmark.get_configuration_space(OptunityConverter())

        tree = search_spaces.SearchTree(conf)
        box = tree.to_box()

        f = logged(self.objective_function)
        f = tree.wrap_decoder(f)
        f = _wrap_hard_box_constraints(f, box, 1)

        suggestion = suggest_solver(self.iterations, "particle swarm", **box)
        solver = make_solver(**suggestion)

        solution, details = optimize(solver, f, maximize=False, max_evals=self.iterations, decoder=tree.decode,
                                     # pmap=map)
                                     pmap=create_pmap(self.n_jobs))

        ls = []
        for meta, value in f.call_log.data.items():
            d = meta._asdict()
            start = d.pop('start', None)
            end = d.pop('end', None)
            ls.append(EvaluationResult(start, end, value, d))

        # Optunity sometimes does not use self.iterations but a little less. Fix number for plotting
        while len(ls) < self.iterations:
            ls.append(ls[-1])

        del f.call_log

        statistics.add_result(ls)
        statistics.stop_optimisation()

        return statistics

    def objective_function(self, **kwargs):
        return self.benchmark.objective_function(kwargs)
