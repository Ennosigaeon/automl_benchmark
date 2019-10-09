import argparse
import warnings
from argparse import Namespace

from hpolib.abstract_benchmark import AbstractBenchmark

import benchmark
import util.logger
from adapter.base import BenchmarkResult
from evaluation.base import Persistence, MongoPersistence


def run(persistence: Persistence, b: AbstractBenchmark, idx: int):
    # db.Branin.drop()
    # db.Branin.find({}, {'solvers.incumbents': 0}).pretty()
    # db.Branin.count()
    # { $where: "this.solvers.length == 0" }

    config_dict = {
        'n_jobs': 8,
        'timeout': None,
        'iterations': 325,
        'seed': idx,

        'random_search': True,
        'grid_search': True,
        'smac': True,
        'hyperopt': False,  # Only single threaded
        'bohb': True,
        'robo': False,  # Only single threaded
        'optunity': True,
        'btb': False  # Only single threaded
    }
    config = Namespace(**config_dict)

    benchmark_result = BenchmarkResult(b, config.n_jobs, config.seed)
    persistence.store_new_run(benchmark_result)

    objective_time = 1

    # Random Search
    if config.random_search:
        from adapter.random_search import ObjectiveRandomSearch
        logger.info('Start random search')
        rs = ObjectiveRandomSearch(config.n_jobs, config.timeout, config.iterations, config.seed)
        stats = rs.optimize(b)
        benchmark_result.add_result(stats)
        persistence.store_results(benchmark_result, stats)

        # Estimate of objective time. Used to select iterations for fixed iterations procedures
        objective_time = stats.runtime['objective_function'][0]

        logger.info('Finished after {}s'.format(stats.end - stats.start))
        logger.info(stats)

    # Grid Search
    if config.grid_search:
        from adapter.grid_search import ObjectiveGridSearch
        logger.info('Start grid search')
        gs = ObjectiveGridSearch(config.n_jobs, config.timeout, config.iterations)
        n = gs.estimate_grid_size(len(b.get_meta_information().get('bounds', [])), objective_time)
        logger.info('Using grid size of {}'.format(n))
        stats = gs.optimize(b, n)
        benchmark_result.add_result(stats)
        persistence.store_results(benchmark_result, stats)
        logger.info('Finished after {}s'.format(stats.end - stats.start))
        logger.info(stats)

    # SMAC
    if config.smac:
        from adapter.smac import SmacAdapter
        logger.info('Start SMAC')
        smac = SmacAdapter(config.n_jobs, config.timeout, config.iterations, config.seed)
        stats = smac.optimize(b, objective_time)
        benchmark_result.add_result(stats)
        persistence.store_results(benchmark_result, stats)
        logger.info('Finished after {}s'.format(stats.end - stats.start))
        logger.info(stats)

    # hyperopt
    if config.hyperopt:
        from adapter.hyperopt_adapter import HyperoptAdapter
        logger.info('Start hyperopt')
        hyperopt = HyperoptAdapter(config.n_jobs, config.timeout, config.iterations, config.seed)
        stats = hyperopt.optimize(b)
        benchmark_result.add_result(stats)
        persistence.store_results(benchmark_result, stats)
        logger.info('Finished after {}s'.format(stats.end - stats.start))
        logger.info(stats)

    # bohb
    if config.bohb:
        from adapter.bohb import BohbAdapter
        logger.info('Start bohb')
        bohb = BohbAdapter(config.n_jobs, config.timeout, config.iterations, config.seed)
        stats = bohb.optimize(b)
        benchmark_result.add_result(stats)
        persistence.store_results(benchmark_result, stats)
        logger.info('Finished after {}s'.format(stats.end - stats.start))
        logger.info(stats)

    # RoBo
    if config.robo:
        from adapter.robo import RoBoAdapter
        logger.info('Start robo')
        robo = RoBoAdapter(config.n_jobs, config.timeout, config.iterations, config.seed)
        stats = robo.optimize(b, model_type='gp')
        benchmark_result.add_result(stats)
        persistence.store_results(benchmark_result, stats)
        logger.info('Finished after {}s'.format(stats.end - stats.start))
        logger.info(stats)

    # Optunity
    if config.optunity:
        from adapter.optunity_adapter import OptunityAdapter
        logger.info('Start optunity')
        optunity = OptunityAdapter(config.n_jobs, config.timeout, config.iterations, config.seed)
        stats = optunity.optimize(b)
        benchmark_result.add_result(stats)
        persistence.store_results(benchmark_result, stats)
        logger.info('Finished after {}s'.format(stats.end - stats.start))
        logger.info(stats)

    # BTB
    if config.btb:
        from adapter.btb_adapter import BtbAdapter
        logger.info('Start btb')
        btb = BtbAdapter(config.n_jobs, config.timeout, config.iterations, config.seed)
        stats = btb.optimize(b)
        benchmark_result.add_result(stats)
        persistence.store_results(benchmark_result, stats)
        logger.info('Finished after {}s'.format(stats.end - stats.start))
        logger.info(stats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, default='localhost')
    parser.add_argument('--chunk', type=int, default=None)
    args = parser.parse_args()

    util.logger.setup(args.chunk)
    logger = util.logger.get()

    warnings.simplefilter(action='ignore', category=FutureWarning)

    logger.info('Main start')
    try:
        persistence = MongoPersistence(url='localhost', db='test')

        task_ids = [11, 12, 14, 16, 18, 20, 21, 22, 23, 28, 31, 32, 36, 37, 43, 45, 49, 53, 58, 219, 2074, 3022,
                    3481, 3485, 3492, 3493, 3494, 3510, 3512, 3549, 3560, 3567, 3573, 3889, 3891, 3896, 3899, 3902,
                    3903, 3913, 3917, 3918, 3954, 9914, 9946, 9950, 9952, 9954, 9955, 9956, 9957, 9960, 9964, 9967,
                    9968, 9970, 9971, 9976, 9977, 9978, 9979, 9980, 9981, 9983, 9985, 9986, 10093, 10101, 14964, 14965,
                    14966, 14969, 14970, 34537, 34539, 125921, 125922, 125923, 146195]
        for task in task_ids:
            logger.info('Starting OpenML benchmark {}'.format(task))
            for i in range(10):
                bm = benchmark.OpenMLBenchmark(task)
                run(persistence, bm, i)
    except (SystemExit, KeyboardInterrupt, Exception) as e:
        logger.error(e, exc_info=True)

    logger.info('Main finished')
