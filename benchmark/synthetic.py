import numpy as np
from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.benchmarks.synthetic_functions import Branin, Bohachevsky, Camelback, Forrester, GoldsteinPrice

from benchmark.base import _dict_as_array
from config import BaseConverter, MetaConfig, NoopConverter


class BraninFunction(Branin):
    @_dict_as_array
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, **kwargs):
        y = (x[1] - (5.1 / (4 * np.pi ** 2)) * x[0] ** 2 + 5 * x[0] / np.pi - 6) ** 2
        y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10

        return {'function_value': y}

    def objective_function_test(self, x, **kwargs):
        return self.objective_function(x)

    @staticmethod
    def get_configuration_space(converter: BaseConverter = NoopConverter()):
        cs = {
            "x1": {
                "type": "uniform_float",
                "lower": -5,
                "upper": 10
            },
            "x2": {
                "type": "uniform_float",
                "lower": 0,
                "upper": 15
            }
        }
        return converter.convert_single(MetaConfig(cs))


class BohachevskyFunction(Bohachevsky):

    @_dict_as_array
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, **kwargs):
        y = 0.7 + x[0] ** 2 + 2.0 * x[1] ** 2
        y -= 0.3 * np.cos(3.0 * np.pi * x[0])
        y -= 0.4 * np.cos(4.0 * np.pi * x[1])

        return {'function_value': y}

    @staticmethod
    def get_configuration_space(converter: BaseConverter = NoopConverter()):
        cs = {
            "x1": {
                "type": "uniform_float",
                "lower": -100,
                "upper": 100
            },
            "x2": {
                "type": "uniform_float",
                "lower": -100,
                "upper": 100
            }
        }
        return converter.convert_single(MetaConfig(cs))


class CamelbackFunction(Camelback):

    @_dict_as_array
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, **kwargs):
        y = (4 - 2.1 * (x[0] ** 2) + ((x[0] ** 4) / 3)) * (x[0] ** 2) + x[0] * x[1] + (-4 + 4 * (x[1] ** 2)) * \
            (x[1] ** 2)
        return {'function_value': y}

    @staticmethod
    def get_configuration_space(converter: BaseConverter = NoopConverter()):
        cs = {
            "x1": {
                "type": "uniform_float",
                "lower": -5,
                "upper": 5
            },
            "x2": {
                "type": "uniform_float",
                "lower": -5,
                "upper": 5
            }
        }
        return converter.convert_single(MetaConfig(cs))


class ForresterFunction(Forrester):

    @_dict_as_array
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, fidelity=1, **kwargs):
        x = x[0]
        y1 = np.power(6 * x - 2, 2) * np.sin(12 * x - 4)

        # best least-squared fit with cubic polynomial
        y2 = 131.09227753 * (x ** 3) - 164.50286816 * (x ** 2) + 50.7228373 * x - 2.84345244
        return {'function_value': fidelity * y1 + (1 - fidelity) * y2, 'cost': fidelity ** 2}

    @staticmethod
    def get_configuration_space(converter: BaseConverter = NoopConverter()):
        cs = {
            "x": {
                "type": "uniform_float",
                "lower": 0,
                "upper": 1
            }
        }
        return converter.convert_single(MetaConfig(cs))


class GoldsteinPriceFunction(GoldsteinPrice):

    @_dict_as_array
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, **kwargs):
        y = (1 + (x[0] + x[1] + 1) ** 2 * (
                19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2)) \
            * (30 + (2 * x[0] - 3 * x[1]) ** 2 * (
                18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2))

        return {'function_value': y}

    @staticmethod
    def get_configuration_space(converter: BaseConverter = NoopConverter()):
        cs = {
            "x1": {
                "type": "uniform_float",
                "lower": -2,
                "upper": 2
            },
            "x2": {
                "type": "uniform_float",
                "lower": -2,
                "upper": 2
            }
        }
        return converter.convert_single(MetaConfig(cs))
