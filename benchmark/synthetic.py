import numpy as np
from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.benchmarks import synthetic_functions

from benchmark.base import _dict_as_array, meta_information
from config import BaseConverter, MetaConfig, NoopConverter


class Bohachevsky(synthetic_functions.Bohachevsky):

    @_dict_as_array
    @AbstractBenchmark._configuration_as_array
    @meta_information
    def objective_function(self, x, **kwargs):
        y = 0.7 + x[0] ** 2 + 2.0 * x[1] ** 2
        y -= 0.3 * np.cos(3.0 * np.pi * x[0])
        y -= 0.4 * np.cos(4.0 * np.pi * x[1])

        return {'function_value': y}

    @staticmethod
    def get_configuration_space(converter: BaseConverter = NoopConverter()):
        cs = {
            "x0": {
                "type": "uniform_float",
                "lower": -100,
                "upper": 100
            },
            "x1": {
                "type": "uniform_float",
                "lower": -100,
                "upper": 100
            }
        }
        return converter.convert_single(MetaConfig(cs))


class Branin(synthetic_functions.Branin):

    @_dict_as_array
    @AbstractBenchmark._configuration_as_array
    @meta_information
    def objective_function(self, x, **kwargs):
        y = (x[1] - (5.1 / (4 * np.pi ** 2)) * x[0] ** 2 + 5 * x[0] / np.pi - 6) ** 2
        y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10

        return {'function_value': y}

    def objective_function_test(self, x, **kwargs):
        return self.objective_function(x)

    @staticmethod
    def get_configuration_space(converter: BaseConverter = NoopConverter()):
        cs = {
            "x0": {
                "type": "uniform_float",
                "lower": -5,
                "upper": 10
            },
            "x1": {
                "type": "uniform_float",
                "lower": 0,
                "upper": 15
            }
        }
        return converter.convert_single(MetaConfig(cs))


class Camelback(synthetic_functions.Camelback):

    @_dict_as_array
    @AbstractBenchmark._configuration_as_array
    @meta_information
    def objective_function(self, x, **kwargs):
        y = (4 - 2.1 * (x[0] ** 2) + ((x[0] ** 4) / 3)) * (x[0] ** 2) + x[0] * x[1] + (-4 + 4 * (x[1] ** 2)) * \
            (x[1] ** 2)
        return {'function_value': y}

    @staticmethod
    def get_configuration_space(converter: BaseConverter = NoopConverter()):
        cs = {
            "x0": {
                "type": "uniform_float",
                "lower": -5,
                "upper": 5
            },
            "x1": {
                "type": "uniform_float",
                "lower": -5,
                "upper": 5
            }
        }
        return converter.convert_single(MetaConfig(cs))


class Forrester(synthetic_functions.Forrester):

    @_dict_as_array
    @AbstractBenchmark._configuration_as_array
    @meta_information
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


class GoldsteinPrice(synthetic_functions.GoldsteinPrice):

    @_dict_as_array
    @AbstractBenchmark._configuration_as_array
    @meta_information
    def objective_function(self, x, **kwargs):
        y = (1 + (x[0] + x[1] + 1) ** 2 * (
                19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2)) \
            * (30 + (2 * x[0] - 3 * x[1]) ** 2 * (
                18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2))

        return {'function_value': y}

    @staticmethod
    def get_configuration_space(converter: BaseConverter = NoopConverter()):
        cs = {
            "x0": {
                "type": "uniform_float",
                "lower": -2,
                "upper": 2
            },
            "x1": {
                "type": "uniform_float",
                "lower": -2,
                "upper": 2
            }
        }
        return converter.convert_single(MetaConfig(cs))


class Hartmann3(synthetic_functions.Hartmann3):

    @_dict_as_array
    @AbstractBenchmark._configuration_as_array
    @meta_information
    def objective_function(self, x, **kwargs):
        external_sum = 0
        for i in range(4):
            internal_sum = 0
            for j in range(3):
                internal_sum += self.A[i, j] * (x[j] - self.P[i, j]) ** 2
            external_sum += self.alpha[i] * np.exp(-internal_sum)

        return {'function_value': -external_sum}

    @staticmethod
    def get_configuration_space(converter: BaseConverter = NoopConverter()):
        cs = {}
        for i in range(3):
            cs[f"x{i}"] = {
                "type": "uniform_float",
                "lower": 0,
                "upper": 1
            }
        return converter.convert_single(MetaConfig(cs))


class Hartmann6(synthetic_functions.Hartmann6):

    @_dict_as_array
    @AbstractBenchmark._configuration_as_array
    @meta_information
    def objective_function(self, x, **kwargs):
        """6d Hartmann test function
            input bounds:  0 <= xi <= 1, i = 1..6
            global optimum: (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573),
            min function value = -3.32237
        """

        external_sum = 0
        for i in range(4):
            internal_sum = 0
            for j in range(6):
                internal_sum += self.A[i, j] * (x[j] - self.P[i, j]) ** 2
            external_sum += self.alpha[i] * np.exp(-internal_sum)

        return {'function_value': -external_sum}

    @staticmethod
    def get_configuration_space(converter: BaseConverter = NoopConverter()):
        cs = {}
        for i in range(6):
            cs[f"x{i}"] = {
                "type": "uniform_float",
                "lower": 0,
                "upper": 1
            }
        return converter.convert_single(MetaConfig(cs))


class Levy(synthetic_functions.Levy):

    @_dict_as_array
    @AbstractBenchmark._configuration_as_array
    @meta_information
    def objective_function(self, x, **kwargs):
        z = 1 + ((x[0] - 1.) / 4.)
        s = np.power((np.sin(np.pi * z)), 2)
        y = (s + ((z - 1) ** 2) * (1 + np.power((np.sin(2 * np.pi * z)), 2)))

        return {'function_value': y}

    @staticmethod
    def get_configuration_space(converter: BaseConverter = NoopConverter()):
        cs = {
            "x": {
                "type": "uniform_float",
                "lower": -15,
                "upper": 10
            }
        }
        return converter.convert_single(MetaConfig(cs))


# Rosenbrock2D, Rosenbrock5D and Rosenbrock10D are not implemented

class Rosenbrock20D(synthetic_functions.rosenbrock.Rosenbrock20D):

    @_dict_as_array
    @AbstractBenchmark._configuration_as_array
    @meta_information
    def objective_function(self, x, **kwargs):
        y = 0
        d = 20
        for i in range(d - 1):
            y += 100 * (x[i + 1] - x[i] ** 2) ** 2
            y += (x[i] - 1) ** 2

        return {'function_value': y}

    @staticmethod
    def get_configuration_space(converter: BaseConverter = NoopConverter()):
        cs = {}
        for i in range(20):
            cs[f"x{i}"] = {
                "type": "uniform_float",
                "lower": 0,
                "upper": 1
            }
        return converter.convert_single(MetaConfig(cs))


class SinOne(synthetic_functions.SinOne):

    @_dict_as_array
    @AbstractBenchmark._configuration_as_array
    @meta_information
    def objective_function(self, x, **kwargs):
        y = 0.5 * np.sin(13 * x[0]) * np.sin(27 * x[0]) + 0.5

        return {'function_value': y}

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


class SinTwo(synthetic_functions.SinTwo):

    @_dict_as_array
    @AbstractBenchmark._configuration_as_array
    @meta_information
    def objective_function(self, x, **kwargs):
        y = (0.5 * np.sin(13 * x[0]) * np.sin(27 * x[0]) + 0.5) * (0.5 * np.sin(13 * x[1]) * np.sin(27 * x[1]) + 0.5)

        return {'function_value': y}

    @staticmethod
    def get_configuration_space(converter: BaseConverter = NoopConverter()):
        cs = {
            "x0": {
                "type": "uniform_float",
                "lower": 0,
                "upper": 1
            },
            "x1": {
                "type": "uniform_float",
                "lower": 0,
                "upper": 1
            }
        }
        return converter.convert_single(MetaConfig(cs))
