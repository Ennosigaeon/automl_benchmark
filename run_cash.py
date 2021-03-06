import argparse
import datetime
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
    # db.getCollectionNames().forEach(function(collname) { db[collname].deleteMany({ $where: "this.solvers.length == 0" }) })

    config_dict = {
        'n_jobs': 3,
        'timeout': None,
        'iterations': 325,
        'seed': idx,

        'random_search': True,
        'grid_search': True,
        'smac': True,
        'hyperopt': True,  # Only single threaded
        'bohb': True,
        'robo': True,  # Only single threaded
        'optunity': True,
        'btb': True  # Only single threaded
    }
    config = Namespace(**config_dict)

    benchmark_result = BenchmarkResult(b, config.n_jobs, config.seed)
    persistence.store_new_run(benchmark_result)

    objective_time = 1

    for old in persistence.load_all(b):
        if old.seed == idx:
            old_results = old
            break
    else:
        old_results = BenchmarkResult(None, -1, -1)

    # Random Search
    if config.random_search:
        from adapter.random_search import ObjectiveRandomSearch
        logger.info('Start random search')
        old = old_results.get_result('Random Search')
        if old is not None:
            logger.info('Reusing old score of {}'.format(old.score))
        else:
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
        old = old_results.get_result('Grid Search')
        if old is not None:
            logger.info('Reusing old score of {}'.format(old.score))
        else:
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
        old = old_results.get_result('SMAC')
        if old is not None:
            logger.info('Reusing old score of {}'.format(old.score))
        else:
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
        old = old_results.get_result('hyperopt')
        if old is not None:
            logger.info('Reusing old score of {}'.format(old.score))
        else:
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
        old = old_results.get_result('BOHB')
        if old is not None:
            logger.info('Reusing old score of {}'.format(old.score))
        else:
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
        old = old_results.get_result('RoBo gp')
        if old is not None:
            logger.info('Reusing old score of {}'.format(old.score))
        else:
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
        old = old_results.get_result('Optunity')
        if old is not None:
            logger.info('Reusing old score of {}'.format(old.score))
        else:
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
        old = old_results.get_result('BTB')
        if old is not None:
            logger.info('Reusing old score of {}'.format(old.score))
        else:
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
        persistence = MongoPersistence(url='localhost', db='tmp')

        task_ids = [9910, 14952, 146817, 146819, 146820, 146824, 167121, 167124, 167125, 167140, 167141]
        for task in task_ids:
            logger.info('#######\nStarting task {}\n#######'.format(task))
            for i in range(10):
                logger.info('##\nIteration {} at {}\n##'.format(i, datetime.datetime.now().time()))
                bm = benchmark.OpenMLBenchmark(task, test_size=None)
                for fold in range(len(bm.folds)):
                    bm.fold = fold
                    run(persistence, bm, i)
    except (SystemExit, KeyboardInterrupt, Exception) as e:
        logger.error(e, exc_info=True)

    logger.info('Main finished')
