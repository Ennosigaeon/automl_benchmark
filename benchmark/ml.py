import importlib
import time

import numpy as np
from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.util import rng_helper
from sklearn import datasets
from sklearn.model_selection import train_test_split

from config import BaseConverter, NoopConverter, MetaConfigCollection


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
        print(conf)
        raise ex


class Iris(AbstractBenchmark):

    def __init__(self, test_size=0.3):
        super().__init__()
        iris = datasets.load_iris()

        X = iris.data
        y = iris.target

        self.train, self.valid, self.train_targets, self.valid_targets = train_test_split(X, y, test_size=test_size)

    def objective_function(self, configuration, dataset_fraction=1, **kwargs):
        start_time = time.time()

        rng = kwargs.get("rng", None)
        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)

        shuffle = self.rng.permutation(self.train.shape[0])
        size = int(dataset_fraction * self.train.shape[0])

        train = self.train[shuffle[:size]]
        train_targets = self.train_targets[shuffle[:size]]

        try:
            clf = create_esimator(configuration)
            clf.fit(train, train_targets)
            y = 1 - clf.score(self.valid, self.valid_targets)
        except Exception as ex:
            print('Uncaught expection {} for {}'.format(ex, configuration))
            y = 1

        c = time.time() - start_time
        return {'function_value': y, 'cost': c, 'start': start_time, 'end': start_time + c}

    def objective_function_test(self, configuration, **kwargs):
        start_time = time.time()

        rng = kwargs.get("rng", None)
        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)

        train = np.concatenate((self.train, self.valid))
        train_targets = np.concatenate((self.train_targets, self.valid_targets))

        try:
            clf = create_esimator(configuration)
            clf.fit(train, train_targets)
            y = 1 - clf.score(self.valid, self.valid_targets)
        except Exception as ex:
            print('Uncaught expection {} for {}'.format(ex, configuration))
            y = 1

        c = time.time() - start_time
        return {'function_value': y, "cost": c}

    @staticmethod
    def get_configuration_space(converter: BaseConverter = NoopConverter()):
        return converter.convert(MetaConfigCollection.from_json('../assets/classifier.json'))

    @staticmethod
    def get_meta_information():
        return {'name': 'Iris', 'cash': True}
