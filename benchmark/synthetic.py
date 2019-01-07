import numpy as np
from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.benchmarks.synthetic_functions import Branin

from benchmark.base import _configuration_as_array
from config import BaseConverter, MetaConfig


class BraninFunction(Branin):
    @_configuration_as_array
    def objective_function(self, x, **kwargs):
        y = (x[1] - (5.1 / (4 * np.pi ** 2)) * x[0] ** 2 + 5 * x[0] / np.pi - 6) ** 2
        y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10

        return {'function_value': y}

    def objective_function_test(self, x, **kwargs):
        return self.objective_function(x)

    @staticmethod
    def get_configuration_space(converter: BaseConverter = None):
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
        if converter is None:
            return cs
        else:
            return converter.convert_single(MetaConfig(cs))

    @staticmethod
    def get_meta_information():
        return {'name': 'Branin',
                'num_function_evals': 100,
                'optima': ([[-np.pi, 12.275],
                            [np.pi, 2.275],
                            [9.42478, 2.475]]),
                'bounds': [[-5, 10], [0, 15]],
                'f_opt': 0.39788735773}
