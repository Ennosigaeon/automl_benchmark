import logging
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
        self.X_valid = None
        self.y_valid = None
        self.X_test = None
        self.y_test = None

        self.logger = logging.getLogger("DataManager")
        self.save_to = os.path.join('/home/vagrant/', "OpenML")
        self.task_id = openml_task_id

        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        if not os.path.isdir(self.save_to):
            self.logger.debug("Create directory {}".format(self.save_to))
            os.makedirs(self.save_to)

        openml.config.apikey = '610344db6388d9ba34f6db45a3cf71de'
        openml.config.set_cache_directory(self.save_to)

    def load(self, test_size: float = 0.3):
        """
        Loads dataset from OpenML in _config.data_directory.
        Downloads data if necessary.

        Returns
        -------
        X_train: np.array
        y_train: np.array
        X_val: np.array
        y_val: np.array
        X_test: np.array
        y_test: np.array
        """

        self.X_train, self.X_test, self.y_train, self.y_test, variable_types, name = self._load_data(self.task_id)
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X_train, self.y_train,
                                                                                  test_size=test_size)

        return self

    def _load_data(self, task_id: int):
        task = openml.tasks.get_task(task_id)

        try:
            task.get_train_test_split_indices(fold=0, repeat=1)
            raise_exception = True
        except:
            raise_exception = False

        if raise_exception:
            logger.fatal(
                'Task {} has more than one repeat. This benchmark can only work with a single repeat.'.format(task_id))

        train_indices, test_indices = task.get_train_test_split_indices()

        X, y = task.get_X_and_y()

        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]

        # TODO replace by more efficient function which only reads in the data
        # saved in the arff file describing the attributes/features
        dataset = task.get_dataset()
        _, _, categorical_indicator = dataset.get_data(
            target=task.target_name,
            return_categorical_indicator=True)
        variable_types = ['categorical' if ci else 'numerical'
                          for ci in categorical_indicator]

        return X_train, X_test, y_train, y_test, variable_types, dataset.name


class OpenMLBenchmark(AbstractBenchmark):

    def __init__(self, task_id: int, test_size: float = 0.3):
        super().__init__()
        self.task_id = task_id

        data = OpenMLHoldoutDataManager(task_id).load(test_size)
        self.X_train = data.X_train
        self.y_train = data.y_train
        self.X_test = data.X_test
        self.y_test = data.y_test
        self.X_valid = data.X_valid
        self.y_valid = data.y_valid

    def objective_function(self, configuration, **kwargs):
        start_time = time.time()

        rng = kwargs.get("rng", None)
        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)

        X_train = self.X_train
        y_train = self.y_train

        try:
            clf = create_esimator(configuration)
            clf.fit(X_train, y_train)
            y = 1 - clf.score(self.X_valid, self.y_valid)
        except Exception as ex:
            logger.error('Uncaught expection {} for {}'.format(ex, configuration))
            y = 1

        c = time.time() - start_time
        return {'function_value': y, 'cost': c, 'start': start_time, 'end': start_time + c}

    def objective_function_test(self, configuration, **kwargs):
        start_time = time.time()

        rng = kwargs.get("rng", None)
        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)

        X_train = np.concatenate((self.X_valid, self.X_train))
        y_train = np.concatenate((self.y_valid, self.y_train))

        try:
            clf = create_esimator(configuration)
            clf.fit(X_train, y_train)
            y = 1 - clf.score(self.X_test, self.y_test)
        except Exception as ex:
            logger.error('Uncaught expection {} for {}'.format(ex, configuration))
            y = 1

        c = time.time() - start_time
        return {'function_value': y, "cost": c}

    @staticmethod
    def get_configuration_space(converter: BaseConverter = NoopConverter()):
        return converter.convert(MetaConfigCollection.from_json('../assets/classifier.json'))

    def get_meta_information(self):
        return {'name': 'OpenML Task {}'.format(self.task_id), 'cash': True}


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
        self.save_to = os.path.join('/home/vagrant/', "OpenML")

        if not os.path.isdir(self.save_to):
            logger.info("Create directory {}".format(self.save_to))
            os.makedirs(self.save_to)

        openml.config.apikey = '610344db6388d9ba34f6db45a3cf71de'
        openml.config.set_cache_directory(self.save_to)

    def load(self) -> Generator[OpenMLBenchmark, None, None]:
        benchmark_suite = openml.study.get_study('OpenML100', 'tasks')

        for task_id in benchmark_suite.tasks:
            if task_id in [34536]:
                logger.debug('Skipping broken OpenML benchmark {}'.format(task_id))
            else:
                logger.debug('Loading OpenML benchmark {}'.format(task_id))
                yield OpenMLBenchmark(task_id)


if __name__ == '__main__':
    util.logger.setup()

    suite = OpenML100Suite()
    ls = []
    for benchmark in suite.load():
        ls.append(benchmark)
    logger.info(len(ls))
