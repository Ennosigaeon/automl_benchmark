import multiprocessing
import time

import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import BOHB
from hpbandster.workers.hpolibbenchmark import HPOlib2Worker
from hpolib.abstract_benchmark import AbstractBenchmark

from adapter.base import BaseAdapter, OptimizationStatistic, EvaluationResult
from config import ConfigSpaceConverter

nameserver = '127.0.0.1'


def start_worker(benchmark: AbstractBenchmark, run_id: str, id: int):
    # noinspection PyArgumentList
    conf = benchmark.get_configuration_space(ConfigSpaceConverter())

    w = HPOlib2Worker(benchmark, conf, nameserver=nameserver, run_id=run_id, id=id)
    w.run(background=False)


class BohbAdapter(BaseAdapter):

    def __init__(self, n_jobs: int, time_limit: float = None, iterations: int = None):
        super().__init__(n_jobs, time_limit, iterations)

    def optimize(self, benchmark: AbstractBenchmark, min_budget: int = 0.1,
                 max_budget: int = 1) -> OptimizationStatistic:
        start = time.time()
        statistics = OptimizationStatistic('BOHB', start, self.n_jobs)

        run_id = '{}_{}'.format(benchmark.get_meta_information()['name'], 0)
        ns = hpns.NameServer(run_id=run_id, host=nameserver, port=None)
        ns.start()

        # noinspection PyArgumentList
        conf = benchmark.get_configuration_space(ConfigSpaceConverter())

        pool = multiprocessing.Pool(processes=self.n_jobs)
        for i in range(self.n_jobs):
            pool.apply_async(start_worker, args=(benchmark, run_id, i), error_callback=self.log_async_error)

        bohb = BOHB(configspace=conf, run_id=run_id, min_budget=min_budget, max_budget=max_budget)
        res = bohb.run(n_iterations=self.iterations, min_n_workers=self.n_jobs)

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
