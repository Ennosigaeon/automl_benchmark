import abc
import math
from abc import ABC, abstractmethod
from importlib import import_module
from typing import Dict

import numpy as np
import scipy.stats
from ConfigSpace import ConfigurationSpace
from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UniformFloatHyperparameter
from hyperopt import hp
from hyperopt.pyll import scope

from config import MetaConfig, MetaConfigCollection, ConfigInheritanceGraph, ConfigFeature, CATEGORICAL, UNI_INT, \
    UNI_FLOAT, PARENT, VALUE


class BaseConverter(ABC):
    @abstractmethod
    def convert(self, config: MetaConfigCollection) -> object:
        pass

    @abstractmethod
    def convert_single(self, config: MetaConfig) -> object:
        pass


class NoopConverter(BaseConverter):

    def convert(self, config: MetaConfigCollection) -> object:
        return config

    def convert_single(self, config: MetaConfig) -> object:
        return config


class ConfigSpaceConverter(BaseConverter):

    def convert(self, config: MetaConfigCollection) -> Dict[str, ConfigurationSpace]:
        '''
        Converting input JSON to SMAC ConfigurationSpace
        :param config: JSON file withe configurations
        :return: ConfigurationSpace
        '''

        from .util import ConfigSpace as util

        configs = {}
        for key, estimator in config.items():
            estimator_cs = self.convert_single(estimator)

            component = util.sklearn_mapping(key)
            component.get_hyperparameter_search_space = lambda dataset_properties=None: estimator_cs

            configs[key] = estimator_cs
        return configs

    def convert_single(self, estimator: MetaConfig) -> ConfigurationSpace:
        '''
        Builds a ConfigurationSpace for a single estimator
        :param estimator: A dict in form
        {parameter_name1: {Type:XY, Min: z1, Max: z2 condition: {parent: p, value: [v]}}parameter_name2 ...}
        :return: ConfigurationSpace for input estimator
        '''
        cs = ConfigurationSpace()

        for name, entry in estimator.items():
            if entry.type == CATEGORICAL:
                cs.add_hyperparameter(
                    CategoricalHyperparameter(name, entry.choices, default_value=entry.default))
            elif entry.type == UNI_INT:
                cs.add_hyperparameter(
                    UniformIntegerHyperparameter(name, entry.lower, entry.upper, default_value=entry.default,
                                                 log=entry.log))
            elif entry.type == UNI_FLOAT:
                cs.add_hyperparameter(
                    UniformFloatHyperparameter(name, entry.lower, entry.upper, default_value=entry.default,
                                               log=entry.log))
            if entry.has_condition():
                cs.add_condition(
                    InCondition(child=cs.get_hyperparameter(name),
                                parent=cs.get_hyperparameter(entry.condition[PARENT]), values=entry.condition[VALUE]))
        return cs


class TpotConverter(BaseConverter):
    def convert(self, config: MetaConfigCollection, points: int = 10) -> dict:
        '''
        Converting input JSON to TPOT config_dict
        :param points: Amount of points a uniform_float should split in to
        :param config: Name of JSON file withe configurations
        :return: config_dict for TPOTClassifier() or TPOTRegressor()
        '''
        config_dict = dict()
        for algorithm, conf in config.items():
            d = self.convert_single(conf, points)
            config_dict[algorithm] = d
        return config_dict

    # noinspection PyMethodOverriding
    def convert_single(self, config: MetaConfig, points: int) -> dict:
        d = dict()
        for key, value in config.items():
            if value.type == CATEGORICAL:
                d[key] = value.choices
            if value.type == UNI_INT:
                d[key] = range(value.lower, value.upper)
            if value.type == UNI_FLOAT:
                steps = abs(value.lower - value.upper) / points
                d[key] = np.arange(value.lower, value.upper, steps)
        return d


class HyperoptConverter(BaseConverter):

    def __init__(self, as_scope: bool = False):
        self.as_scope = as_scope

    def convert(self, config: MetaConfigCollection) -> hp.choice:
        '''
        Converting input JSON to Hyperopt ConfigurationSpace
        :param config: JSON file withe configurations
        :param as_scope:
        :return: ConfigurationSpace
        '''
        config_space = []
        for key, conf in config.items():
            d = self.convert_single(conf, key)
            config_space.append(d)
        return hp.choice('estimator_type', config_space)

    # noinspection PyMethodOverriding
    def convert_single(self, config: MetaConfig, algorithm: str = '') -> dict:
        parents = set()
        for key, param in config.items():
            if param.has_condition():
                parents.add(param.condition['parent'])

        if len(parents) > 1:
            raise ValueError('More than one parent is currently no supported')

        for parent in parents:
            label = 'custom_{}'.format(algorithm)
            c = config.dict[parent]

            if c.type != CATEGORICAL:
                raise ValueError('Non categorical parameter has children')
            l = [self.__get_algo_config(config, algorithm, parent, choice) for choice in c.choices]
            return hp.choice(label, l)

        return self.__get_algo_config(config, algorithm)

    def __get_algo_config(self, config: MetaConfig, algorithm: str, parent: str = None, parent_value: str = None):
        d = {}
        for parameter, value in config.items():
            if self.as_scope:
                label = 'custom_{}_{}_{}'.format(algorithm, parent_value if parent_value is not None else '', parameter)
            else:
                label = parameter

            if parameter == parent:
                d[parameter] = parent_value
            else:
                if value.has_condition() and value.condition['parent'] == parent and parent_value not in \
                        value.condition['value']:
                    continue

                if value.type == UNI_INT:
                    # TODO check if difference between hyperopt and hyperopt-sklearn
                    # d[parameter] = hp.quniform(label, value.lower, value.upper, 1)
                    d[parameter] = scope.int(hp.quniform(label, value.lower, value.upper, 1))
                elif value.type == UNI_FLOAT:
                    d[parameter] = hp.uniform(label, value.lower, value.upper)
                elif value.type == CATEGORICAL:
                    d[parameter] = hp.choice(label, value.choices)

        if self.as_scope:
            return scope.generate_sklearn_estimator(algorithm, **d)
        else:
            return d

    @staticmethod
    @scope.define
    def generate_sklearn_estimator(estimator_name, *args, **kwargs):
        module_name = estimator_name.rpartition('.')[0]
        class_name = estimator_name.split('.')[-1]
        module = import_module(module_name)
        class_ = getattr(module, class_name)
        return class_(*args, **kwargs)


class BtbConverter(BaseConverter):
    def convert(self, config: MetaConfigCollection) -> list:
        ls = []
        for algorithm, conf in config.items():
            ls.append(self.convert_single(conf, algorithm))
        return ls

    # noinspection PyMethodOverriding
    def convert_single(self, config: MetaConfig, name: str = 'foo') -> dict:
        hyperparamters = {}
        root = []
        conditional = {}
        for key, value in config.items():
            if value.type == CATEGORICAL:
                t = 'bool' if value.choices[0] in [True, False] else 'string'
                hyperparamters[key] = {'type': t, 'values': value.choices}
            if value.type == UNI_INT:
                hyperparamters[key] = {'type': 'int', 'range': [value.lower, value.upper]}
            if value.type == UNI_FLOAT:
                hyperparamters[key] = {'type': 'float', 'range': [value.lower, value.upper]}

            if value.condition is None:
                root.append(key)
            else:
                d = conditional.setdefault(value.condition['parent'], {})
                for v in value.condition['value']:
                    d.setdefault(v, []).append(key)

        d = {
            'name': name,
            'class': name,
            'hyperparameters': hyperparamters,
            'root_hyperparameters': root,
            'conditional_hyperparameters': conditional
        }
        return d


class NaiveSearchConverter(BaseConverter, abc.ABC):

    def __init__(self):
        self.processed_nodes = set()

    def convert(self, config: MetaConfigCollection):
        estimators = {}
        for name, conf in config.items():
            d = self.convert_single(conf)
            estimators.update({name: d})
        return estimators

    def convert_single(self, conf: MetaConfig) -> dict:
        d = {}
        self.processed_nodes = set()
        graph = ConfigInheritanceGraph(conf)

        for key, value in conf.items():
            if key in self.processed_nodes:
                continue
            d.update({key: self._get_algo_config(key, value, graph)})
        return d

    @abc.abstractmethod
    def _get_algo_config(self, key, value: ConfigFeature, graph: ConfigInheritanceGraph):
        pass


class RandomSearchConverter(NaiveSearchConverter):

    def _get_algo_config(self, key, value: ConfigFeature, graph: ConfigInheritanceGraph):
        self.processed_nodes.add(key)

        if value.type == UNI_INT:
            return range(value.lower, value.upper)
        elif value.type == UNI_FLOAT:
            return scipy.stats.uniform(loc=value.lower, scale=value.upper)
        elif value.type == CATEGORICAL:
            choices_list = []
            for choice in value.choices:
                choices_list.append(choice)
            return choices_list
        else:
            raise ValueError('Unknown type {}'.format(value.type))


class GridSearchConverter(NaiveSearchConverter):

    def __init__(self, n: int = 10):
        super().__init__()
        self.n = n

    def _get_algo_config(self, key, value: ConfigFeature, graph: ConfigInheritanceGraph):
        self.processed_nodes.add(key)

        if value.type == UNI_INT:
            if value.lower == value.upper:
                return [value.lower]

            size = abs(value.lower - value.upper)
            if size <= self.n:
                return np.arange(value.lower, value.upper, 1)
            else:
                return np.arange(value.lower, value.upper, math.ceil(size / self.n))
        elif value.type == UNI_FLOAT:
            if value.lower == value.upper:
                return [value.lower]
            return np.linspace(value.lower, value.upper, self.n)
        elif value.type == CATEGORICAL:
            choices_list = []
            for choice in value.choices:
                choices_list.append(choice)
            return choices_list
        else:
            raise ValueError('Unknown type {}'.format(value.type))


class RoBoConverter(BaseConverter):
    def convert(self, config: MetaConfigCollection) -> object:
        raise NotImplementedError('RoBo is not suited for CASH solving')

    def convert_single(self, config: MetaConfig) -> object:
        lower = []
        upper = []
        names = []

        for name, value in config.items():
            if value.type == UNI_FLOAT:
                lower.append(value.lower)
                upper.append(value.upper)
                names.append(name)
            else:
                raise ValueError('RoBo can only handle floating numbers')

        return np.array(lower), np.array(upper), names


class OptunityConverter(BaseConverter):
    def convert(self, config: MetaConfigCollection) -> object:
        d = {}
        for key, conf in config.items():
            d[key] = self.convert_single(conf)
        return {'algorithm': d}

    def convert_single(self, config: MetaConfig) -> object:
        parents = set()
        for key, param in config.items():
            if param.has_condition():
                parents.add(param.condition['parent'])

        if len(parents) > 1:
            raise ValueError('More than one parent is currently no supported')

        for parent in parents:
            c = config.dict[parent]

            if c.type != CATEGORICAL:
                raise ValueError('Non categorical parameter has children')

            d = {}
            for choice in c.choices:
                d[choice] = self.__get_algo_config(config, parent, choice)
            return {parent: d}

        return self.__get_algo_config(config)

    @staticmethod
    def __get_algo_config(config: MetaConfig, parent: str = None, parent_value: str = None):
        d = {}
        for parameter, value in config.items():
            if parameter == parent:
                continue
            if value.has_condition() and value.condition['parent'] == parent and parent_value not in \
                    value.condition['value']:
                continue

            if value.type == UNI_INT:
                tmp = {}
                for i in range(value.lower, value.upper):
                    tmp[str(i)] = None
                d[parameter] = tmp
            elif value.type == UNI_FLOAT:
                d[parameter] = [value.lower, value.upper]
            elif value.type == CATEGORICAL:
                tmp = {}
                for c in value.choices:
                    tmp[str(c)] = None
                d[parameter] = tmp

        return d


class SpearmintConverter(BaseConverter):
    def convert(self, config: MetaConfigCollection) -> object:
        d = {}
        for key, conf in config.items():
            d[key] = self.convert_single(conf, algorithm=key)
        return d

    def convert_single(self, config: MetaConfig, algorithm: str = 'foo') -> object:
        variables = {}

        for name, value in config.items():
            if value.type == UNI_FLOAT:
                variables[name] = {
                    'type': 'FLOAT',
                    'size': 1,
                    'min': value.lower,
                    'max': value.upper
                }
            if value.type == UNI_INT:
                variables[name] = {
                    'type': 'INT',
                    'size': 1,
                    'min': value.lower,
                    'max': value.upper
                }
            if value.type == CATEGORICAL:
                variables[name] = {
                    'type': 'ENUM',
                    'size': 1,
                    'options': value.choices
                }

        return {
            'experiment-name': algorithm,
            'likelihood': 'NOISELESS',
            'variables': variables
        }
