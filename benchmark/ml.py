import importlib
import time

import numpy as np
from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.util import rng_helper
from sklearn import datasets
from sklearn.model_selection import train_test_split

import util.logger
from config import BaseConverter, NoopConverter, MetaConfigCollection

logger = util.logger.get()


def create_esimator(conf: dict):
    try:
        name = conf['algorithm']
        kwargs = conf.copy()
        del kwargs['algorithm']

        module_name = name.rpartition(".")[0]
        class_name = name.split(".")[-1]

        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
        return class_(**kwargs)
    except Exception as ex:
        logger.error('Invalid estimator with config {}'.format(conf))
        raise ex


class Iris(AbstractBenchmark):

    def __init__(self, test_size=0.3):
        super().__init__()
        iris = datasets.load_iris()

        X = iris.data
        y = iris.target

        self.X_train, self.y_train, self.X_test, self.y_test = train_test_split(X, y, test_size=test_size)
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X_train, self.y_train,
                                                                                  test_size=test_size)

    def objective_function(self, configuration, dataset_fraction=1, **kwargs):
        start_time = time.time()

        rng = kwargs.get("rng", None)
        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)

        shuffle = self.rng.permutation(self.X_train.shape[0])
        size = int(dataset_fraction * self.X_train.shape[0])

        X_train = self.X_train[shuffle[:size]]
        y_train = self.y_train[shuffle[:size]]

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

        X_train = np.concatenate((self.X_train, self.X_valid))
        y_train = np.concatenate((self.y_train, self.y_valid))

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

    @staticmethod
    def get_meta_information():
        return {'name': 'Iris', 'cash': True}
