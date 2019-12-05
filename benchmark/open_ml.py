import multiprocessing
import os
import time
from typing import Generator, Optional

import math
import numpy as np
import openml
import pandas as pd
from hpolib.util import rng_helper
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder

import util.logger
from benchmark import AbstractBenchmark, create_esimator
from config import BaseConverter, NoopConverter, MetaConfigCollection

logger = util.logger.get()


class OpenMLDataManager():
    def __init__(self, openml_task_id: int, rng=None):
        self.X = None
        self.y = None
        self.categorical = None
        self.names = None
        self.folds = []

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

    def load(self) -> 'OpenMLDataManager':
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
        X, y, categorical, self.names = dataset.get_data(
            target=dataset.default_target_attribute
        )

        for name, cat in zip(self.names, categorical):
            if cat:
                enc = LabelEncoder()
                missing = np.any(pd.isna(X[name]))

                missing_vec = pd.isna(X[name])

                x_tmp = X[name].cat.add_categories(['<MISSING>']).fillna('<MISSING>')
                X[name] = enc.fit_transform(x_tmp)

                if missing:
                    idx = enc.transform(['<MISSING>'])[0]
                    X[name][X[name] == idx] = np.nan
                    assert pd.isna(X[name]).equals(missing_vec)

        X = X.values
        y = y.values.__array__()
        self.y = LabelEncoder().fit_transform(y)
        self.X = X.astype(np.float64)
        self.categorical = categorical
        return self


class OpenMLHoldoutDataManager(OpenMLDataManager):

    def load(self, test_size: float = 0.3) -> 'OpenMLHoldoutDataManager':
        super().load()
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size)
        ls = [X_train, y_train, X_test, y_test]
        self.folds.append(ls)
        return self


class OpenMLCVDataManager(OpenMLDataManager):

    def load(self, n_splits: int = 4) -> 'OpenMLCVDataManager':
        super().load()
        kf = KFold(n_splits=n_splits)
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            ls = [X_train, y_train, X_test, y_test]
            self.folds.append(ls)
        return self


class OpenMLBenchmark(AbstractBenchmark):

    def __init__(self, task_id: int, test_size: Optional[float] = 0.3, n_folds: Optional[int] = 4, load: bool = True):
        super().__init__()
        self.task_id = task_id

        if load:
            if test_size is not None:
                data = OpenMLHoldoutDataManager(task_id).load(test_size)
            else:
                data = OpenMLCVDataManager(task_id).load(n_folds)
            self.folds = data.folds
            self.categorical = data.categorical
            self.column_names = data.names

    def objective_function(self, configuration, timeout: int = 300, budget=1, seed=None):
        start_time = time.time()
        manager = multiprocessing.Manager()
        score = manager.Value('d', 1.0)
        avg_score = 0

        # logger.debug('Testing configuration {}'.format(configuration))
        for fold in self.folds:
            X_train, y_train, X_test, y_test = fold
            self.rng = rng_helper.get_rng(rng=seed, self_rng=self.rng)

            shuffle = self.rng.permutation(X_train.shape[0])
            size = int(budget * X_train.shape[0])

            X_train = X_train[shuffle[:size]]
            y_train = y_train[shuffle[:size]]

            p = multiprocessing.Process(target=self._fit_and_score, args=(configuration, X_train, y_train, score))
            p.start()
            p.join(30)

            if p.is_alive():
                logger.debug('Abort fitting after timeout')
                p.terminate()
                p.join()
            avg_score += score.value

        c = time.time() - start_time
        return {'function_value': avg_score / len(self.folds), 'cost': c, 'start': start_time, 'end': start_time + c}

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
        print(len(benchmark.X_test) + len(benchmark.X_train))
        ls.append(benchmark)
    logger.info(len(ls))
