import abc
import math
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import scipy.stats
from ConfigSpace import ConfigurationSpace
from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UniformFloatHyperparameter
from hyperopt import hp

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
        """
        Converting input JSON to SMAC ConfigurationSpace
        :param config: JSON file withe configurations
        :return: ConfigurationSpace
        """

        from .util import ConfigSpace as util

        configs = {}
        for key, estimator in config.items():
            estimator_cs = self.convert_single(estimator)

            component = util.sklearn_mapping(key)
            component.get_hyperparameter_search_space = lambda dataset_properties=None: estimator_cs

            configs[key] = estimator_cs
        return configs

    def convert_single(self, estimator: MetaConfig) -> ConfigurationSpace:
        """
        Builds a ConfigurationSpace for a single estimator
        :param estimator: A dict in form
        {parameter_name1: {Type:XY, Min: z1, Max: z2 condition: {parent: p, value: [v]}}parameter_name2 ...}
        :return: ConfigurationSpace for input estimator
        """
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
        """
        Converting input JSON to TPOT config_dict
        :param points: Amount of points a uniform_float should split in to
        :param config: Name of JSON file withe configurations
        :return: config_dict for TPOTClassifier() or TPOTRegressor()
        """
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

    def convert(self, config: MetaConfigCollection) -> hp.choice:
        """
        Converting input JSON to Hyperopt ConfigurationSpace
        :param config: JSON file withe configurations
        :return: ConfigurationSpace
        """
        config_space = []
        for key, conf in config.items():
            d = self.convert_single(conf, key)
            config_space.append(d)
        return hp.choice('estimator_type', config_space)

    # noinspection PyMethodOverriding
    def convert_single(self, config: MetaConfig, key: str) -> dict:
        d = {'type': key}
        graph = ConfigInheritanceGraph(config)
        for child in list(graph.successors(graph.ROOT)):
            d.update(self.__get_algo_config(key, child, graph))
        return d

    def __get_algo_config(self, parent: str, parameter: str, graph: ConfigInheritanceGraph) -> dict:
        """
        Builds a nested ConfigurationDict for a estimator and its child parameters by recursion
        :param parent: String with name of parent parameter
        :param parameter: String withe name of actual parameter
        :param graph: Directed graph representing inheritance of parameters
        :return: ConfigurationDict for input estimator
        """
        config: ConfigFeature = graph.get_config()[parameter]
        label = parent + "_" + parameter
        if config.type == UNI_INT:
            return {parameter: hp.quniform(label, config.lower, config.upper, 1)}
        elif config.type == UNI_FLOAT:
            return {parameter: hp.uniform(label, config.lower, config.upper)}
        elif config.type == CATEGORICAL:
            type_label = "option"
            choices_list = []
            for choice in config.choices:
                children = graph.edge_dfs(choice)
                result = {type_label: choice}

                if (len(children) == 1):
                    from_node = children[0][0]
                    to_node = children[0][1]
                    result = {type_label: self.__get_algo_config(from_node, to_node, graph)[to_node]}
                else:
                    for child in graph.successors(choice):
                        result.update(self.__get_algo_config(choice, child, graph))
                choices_list.append(result)
            return {parameter: hp.choice(label, choices_list)}


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
                children = graph.edge_dfs(choice)

                if len(children) == 1:
                    c = graph.get_config()[children[0][1]]
                    choices_list.append(self._get_algo_config(children[0][1], c, graph))
                else:
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
                children = graph.edge_dfs(choice)

                if len(children) == 1:
                    c = graph.get_config()[children[0][1]]
                    choices_list.extend(self._get_algo_config(children[0][1], c, graph))
                else:
                    choices_list.append(choice)
            return choices_list
        else:
            raise ValueError('Unknown type {}'.format(value.type))
