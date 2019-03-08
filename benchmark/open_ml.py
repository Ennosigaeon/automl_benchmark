import math
import multiprocessing
import os
import time
from typing import Generator

import numpy as np
import openml
from hpolib.util import rng_helper
from sklearn.model_selection import train_test_split

import util.logger
from benchmark import AbstractBenchmark, create_esimator
from config import BaseConverter, NoopConverter, MetaConfigCollection

logger = util.logger.get()


class OpenMLHoldoutDataManager():
    def __init__(self, openml_task_id: int, rng=None):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.save_to = os.path.expanduser('~/OpenML')
        self.task_id = openml_task_id

        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        if not os.path.isdir(self.save_to):
            logger.debug('Create directory {}'.format(self.save_to))
            os.makedirs(self.save_to)

        openml.config.apikey = '610344db6388d9ba34f6db45a3cf71de'
        openml.config.set_cache_directory(self.save_to)

    def load(self, test_size: float = 0.3):
        '''
        Loads dataset from OpenML in _config.data_directory.
        Downloads data if necessary.

        Returns
        -------
        X_train: np.array
        y_train: np.array
        X_test: np.array
        y_test: np.array
        '''

        task = openml.tasks.get_task(self.task_id)

        dataset = openml.datasets.get_dataset(dataset_id=task.dataset_id)
        X, y = dataset.get_data(
            target=dataset.default_target_attribute,
            return_attribute_names=False,
            return_categorical_indicator=False
        )

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size)
        return self


class OpenMLBenchmark(AbstractBenchmark):

    def __init__(self, task_id: int, test_size: float = 0.3, load: bool = True):
        super().__init__()
        self.task_id = task_id

        if load:
            data = OpenMLHoldoutDataManager(task_id).load(test_size)
            self.X_train = data.X_train
            self.y_train = data.y_train
            self.X_test = data.X_test
            self.y_test = data.y_test

    def objective_function(self, configuration, timeout: int = 300, budget=1, seed=None):
        start_time = time.time()
        manager = multiprocessing.Manager()
        score = manager.Value('d', 1.0)

        logger.debug('Testing configuration {}'.format(configuration))

        self.rng = rng_helper.get_rng(rng=seed, self_rng=self.rng)

        shuffle = self.rng.permutation(self.X_train.shape[0])
        size = int(budget * self.X_train.shape[0])

        X_train = self.X_train[shuffle[:size]]
        y_train = self.y_train[shuffle[:size]]

        p = multiprocessing.Process(target=self._fit_and_score, args=(configuration, X_train, y_train, score))
        p.start()
        p.join(timeout)

        if p.is_alive():
            logger.debug('Abort fitting after timeout')
            p.terminate()
            p.join()

        c = time.time() - start_time
        return {'function_value': score.value, 'cost': c, 'start': start_time, 'end': start_time + c}

    def _fit_and_score(self, configuration, X_train, y_train, score):
        try:
            clf = create_esimator(configuration)
            clf = clf.fit(X_train, y_train)
            score.value = 1 - clf.score(self.X_test, self.y_test)
        except Exception as ex:
            logger.error('Uncaught exception {} for {}'.format(ex, configuration))

    def objective_function_test(self, configuration, **kwargs):
        start_time = time.time()

        rng = kwargs.get('rng', None)
        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)

        try:
            clf = create_esimator(configuration)
            clf.fit(self.X_train, self.y_train)
            y = 1 - clf.score(self.X_test, self.y_test)
        except Exception as ex:
            logger.error('Uncaught expection {} for {}'.format(ex, configuration))
            y = 1

        c = time.time() - start_time
        return {'function_value': y, 'cost': c}

    @staticmethod
    def get_configuration_space(converter: BaseConverter = NoopConverter()):
        return converter.convert(MetaConfigCollection.from_json('assets/classifier.json'))

    def get_meta_information(self):
        return {'name': 'OpenML_Task_{}'.format(self.task_id), 'cash': True}


def fix_no_tags(result_dict, tag):
    v = result_dict.get(tag, [])
    if isinstance(v, list):
        return v
    elif isinstance(v, dict):
        return [v]
    else:
        raise TypeError()


openml.study.functions._multitag_to_list = fix_no_tags


class OpenML100Suite:

    def __init__(self):
        self.save_to = os.path.expanduser('~/OpenML')

        if not os.path.isdir(self.save_to):
            logger.info('Create directory {}'.format(self.save_to))
            os.makedirs(self.save_to)

        openml.config.apikey = '610344db6388d9ba34f6db45a3cf71de'
        openml.config.set_cache_directory(self.save_to)

    @staticmethod
    def load(chunk: int = None, total_chunks: int = 8) -> Generator[OpenMLBenchmark, None, None]:
        benchmark_suite = openml.study.get_study('OpenML100', 'tasks')
        chunk_size = int(math.ceil(len(benchmark_suite.tasks) / total_chunks))

        for i, task_id in enumerate(benchmark_suite.tasks):
            if chunk is not None and (i < chunk * chunk_size or i >= (chunk + 1) * chunk_size):
                continue

            if task_id in [34536]:
                logger.info('Skipping broken OpenML benchmark {}'.format(task_id))
            elif task_id in [22, 3481, 3573]:
                logger.info('Skipping long running OpenML benchmark {}'.format(task_id))
            else:
                logger.debug('Loading OpenML benchmark {}'.format(task_id))
                yield OpenMLBenchmark(task_id)

    @staticmethod
    def tasks():
        benchmark_suite = openml.study.get_study('OpenML100', 'tasks')
        return benchmark_suite.tasks


if __name__ == '__main__':
    util.logger.setup()

    suite = OpenML100Suite()
    ls = []
    for benchmark in suite.load():
        ls.append(benchmark)
    logger.info(len(ls))
