import time

import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import BOHB
from hpbandster.workers.hpolibbenchmark import Worker
from hpolib.abstract_benchmark import AbstractBenchmark

from adapter.base import BaseAdapter, OptimizationStatistic, EvaluationResult
from config import ConfigSpaceConverter
from util.multiprocessor import NoDaemonPool

nameserver = '127.0.0.1'


def start_worker(benchmark: AbstractBenchmark, run_id: str, id: int):
    # noinspection PyArgumentList
    conf = benchmark.get_configuration_space(ConfigSpaceConverter())

    w = HPOlib2Worker(benchmark, configspace=conf, nameserver=nameserver, run_id=run_id, id=id, config_as_array=False)
    w.run(background=False)


class BohbAdapter(BaseAdapter):

    def __init__(self, n_jobs: int, time_limit: float = None, iterations: int = None, seed: int = None):
        super().__init__(n_jobs, time_limit, iterations, seed)

    def optimize(self, benchmark: AbstractBenchmark, min_budget: int = 0.1,
                 max_budget: int = 1) -> OptimizationStatistic:
        start = time.time()
        statistics = OptimizationStatistic('BOHB', start)

        run_id = '{}_{}'.format(benchmark.get_meta_information()['name'], 0)
        ns = hpns.NameServer(run_id=run_id, host=nameserver, port=None)
        ns.start()

        # noinspection PyArgumentList
        conf = benchmark.get_configuration_space(ConfigSpaceConverter())

        pool = NoDaemonPool(processes=self.n_jobs)
        for i in range(self.n_jobs):
            pool.apply_async(start_worker, args=(benchmark, run_id, i), error_callback=self.log_async_error)

        bohb = BOHB(configspace=conf, run_id=run_id, min_budget=min_budget, max_budget=max_budget)
        # Fix number of iterations, such that in total self.iterations objective function is called
        n = (self.iterations * 0.9) / 6
        res = bohb.run(n_iterations=n, min_n_workers=self.n_jobs)

        bohb.shutdown(shutdown_workers=True)
        ns.shutdown()

        pool.close()
        pool.join()

        configs = res.get_id2config_mapping()
        ls = []
        for run in res.get_all_runs():
            ls.append(EvaluationResult.from_dict(run.info, configs[run.config_id]['config']))
        statistics.add_result(ls)
        statistics.stop_optimisation()

        return statistics


class HPOlib2Worker(Worker):
    def __init__(self, benchmark, configspace=None, budget_name='budget', budget_preprocessor=None,
                 config_as_array=True, **kwargs):

        super().__init__(**kwargs)
        self.benchmark = benchmark

        if configspace is None:
            self.configspace = benchmark.get_configuration_space()
        else:
            self.configspace = configspace

        self.budget_name = budget_name

        if budget_preprocessor is None:
            self.budget_preprocessor = lambda b: b
        else:
            self.budget_preprocessor = budget_preprocessor

        self.config_as_array = config_as_array

    def compute(self, config, budget, **kwargs):
        c = {}

        algorithm = config.get('__choice__', '')
        if len(algorithm) > 0:
            n = len(algorithm) + 1
            c['algorithm'] = algorithm
        else:
            n = 0

        for key, value in config.items():
            if key == '__choice__':
                continue
            c[key[n:]] = value

        kwargs = {self.budget_name: self.budget_preprocessor(budget)}
        res = self.benchmark.objective_function(c, **kwargs)
        return ({
            'loss': res['function_value'],
            'info': res
        })
