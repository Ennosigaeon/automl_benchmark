import inspect
import json
import os
import subprocess
import time

from hpolib.abstract_benchmark import AbstractBenchmark

from adapter.base import BaseAdapter, OptimizationStatistic
from config import SpearmintConverter


class SpearmintAdapter(BaseAdapter):

    def __init__(self, n_jobs: int, time_limit: float = None, iterations: int = None, seed: int = None):
        super().__init__(n_jobs, time_limit, iterations, seed)
        self.benchmark = None

        if self.iterations is None:
            raise NotImplementedError('Timeout not supported yet')

    # noinspection PyMethodOverriding
    def optimize(self, benchmark: AbstractBenchmark):
        self.benchmark = benchmark

        start = time.time()
        statistics = OptimizationStatistic('Spearmint', start)

        config, file = self._get_config(benchmark)

        cmd = 'python3 /vagrant/phd/exisiting_solutions/CASH/Spearmint/spearmint/main.py /tmp --config {}'.format(file)
        print(cmd)

        # subprocess.check_call(cmd, shell=True)

        statistics.stop_optimisation()
        return statistics

    def _get_config(self, benchmark: AbstractBenchmark):
        # noinspection PyArgumentList
        config = benchmark.get_configuration_space(SpearmintConverter())
        name = '{}-{}'.format(benchmark.get_meta_information()['name'], self.seed)

        config['max-finished-jobs'] = self.iterations
        config['max-concurrent'] = self.n_jobs
        config['experiment-name'] = name
        config['main-file'] = '{}/spearmint_main.py'.format(os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe()))))
        config['database'] = {'address': '10.0.2.2'}
        config['variables']['__benchmark__'] = {
            'type': 'ENUM',
            'size': 1,
            'options': [benchmark.get_meta_information()['name']]
        }

        n = '/tmp/{}.json'.format(name)
        with open(n, 'w') as f:
            json.dump(config, f)
        return config, n